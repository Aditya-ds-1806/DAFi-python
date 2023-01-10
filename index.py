# %%
import numpy as np
from librosa import load
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal.windows import hann
from pymongo import MongoClient
import glob
import json

# %%
DATABASE_URI = (
    "mongodb+srv://test:test@dafi.3uhsqyq.mongodb.net/?retryWrites=true&w=majority"
)
rcParams["figure.dpi"] = 100

# %%
client = MongoClient(DATABASE_URI)
db = client.DAFi
pairs_collection = "pairs"
song_info_collection = "info"

if pairs_collection not in db.list_collection_names():
    db.create_collection(pairs_collection)
if song_info_collection not in db.list_collection_names():
    db.create_collection(song_info_collection)

# %%
fs = 44100
win_size = 1024
anchor_offset = 3
target_zone_size = 5
target_zone_coeff = 0.6
precision = 6
time_res = win_size / fs
build_fingerprint_db = False
build_metadata_db = False

# %%
song_info = [
    {
        "song_id": 1,
        "song_name": "Aulia Tere Daman Laagi",
        "artist": "Shankar Mahadevan",
    },
    {
        "song_id": 2,
        "song_name": "Mangal Bhavan Amangal Haari",
        "artist": "Shankar Mahadevan",
    },
    {"song_id": 3, "song_name": "Jana Gana Mana", "artist": "Rabindranath Tagore"},
    {"song_id": 4, "song_name": "Vande Mataram", "artist": "Bankim Chandra Chatterjee"},
    {
        "song_id": 5,
        "song_name": "Ye Zameen meri mera aasmaan",
        "artist": ["Gulzar sab", "Anuradha Sharma"],
    },
]

if build_metadata_db:
    db.info.insert_many(song_info)

# %%
def compute_spectrogram(audio_path, plot_spec=False, plot_filtered_spec=False):
    samples, _ = load(audio_path, sr=fs, mono=True)
    freq, time, fft_values = spectrogram(
        samples,
        fs=fs,
        window=hann(win_size),
        noverlap=0,
        nfft=win_size,
        detrend=False,
        mode="magnitude",
        nperseg=win_size,
    )
    if plot_spec:
        plt.figure()
        plt.pcolormesh(
            time, freq / 1000, 20 * np.log10(fft_values, dtype=float), cmap="BrBG"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (kHz)")
        plt.title("Audio Spectrogram")
        plt.colorbar()
        plt.show()

    filtered_spectrogram = filter_spectrogram(
        fft_values, time.size, plot=plot_filtered_spec
    )

    return filtered_spectrogram


# %%
def filter_spectrogram(fft_values, time_slices, plot=False):
    # logarithmic_bands = [(0,10), (10,20), (20,40), (40,80), (80,160), (160,512)]
    logarithmic_bands = [(0, 20), (20, 60), (60, 120), (120, 240), (240, 512)]
    nbands = len(logarithmic_bands)
    filtered_spectrogram = {}
    slice_avgs = np.zeros((time_slices, nbands))

    for time_slice in range(time_slices):
        temp = []
        slice_avg = 0

        for i, band in enumerate(logarithmic_bands):
            fft_slice = fft_values[band[0] : band[1], time_slice]
            # find strongest frequency point in band
            idx = np.argmax(fft_slice)
            slice_avgs[time_slice][i] = fft_slice[idx]
            slice_avg += fft_slice[idx]
            temp.append((time_slice, band[0] + idx, fft_slice[idx], i))

        slice_avg /= nbands

        # Keep points stronger than the slice avg
        for (time_slice, freq_bin, mag, band_num) in temp:
            if mag >= slice_avg:
                if time_slice not in filtered_spectrogram:
                    filtered_spectrogram[time_slice] = []
                filtered_spectrogram[time_slice].append((freq_bin, mag, band_num))

    slice_avgs = np.mean(slice_avgs, axis=0)

    # Drop points weaker than the mean in band across entire song
    for time_slice, points in filtered_spectrogram.items():
        points = [
            freq_bin
            for (freq_bin, _, _) in list(
                filter(lambda pt: pt[1] >= slice_avgs[pt[2]], points)
            )
        ]
        filtered_spectrogram[time_slice] = points

    filtered_spectrogram = dict(
        filter(lambda couple: len(couple[1]) > 0, filtered_spectrogram.items())
    )

    if plot:
        plt.figure()
        for slice, points in filtered_spectrogram.items():
            plt.scatter(
                slice * np.ones(len(points)) * time_res, points, marker="x", c="r"
            )

        for band in logarithmic_bands:
            plt.axhline(y=band[0], color="k", linestyle="-")

        plt.ylim(0, win_size / 2)
        plt.xlabel("Time(s)")
        plt.ylabel("Frequency bins")
        plt.title("Filtered Spectrogram")
        plt.show()

    return filtered_spectrogram


# %%
def hash_song(filtered_spectrogram, songId=None, insert=False):
    pairs = {}
    sorted_points = []

    for time in sorted(filtered_spectrogram.keys()):
        for freq in sorted(filtered_spectrogram[time]):
            sorted_points.append((time, freq))

    total_pts = len(sorted_points)
    t_ap, f_ap = sorted_points[0][0], sorted_points[0][1]
    last_ap = total_pts - target_zone_size - anchor_offset + 1

    for ap in range(last_ap):
        t_ap, f_ap = sorted_points[ap][0], sorted_points[ap][1]
        target_zone = (ap + anchor_offset, ap + anchor_offset + target_zone_size)
        for target_pt in range(target_zone[0], target_zone[1]):
            t_tp, f_tp = sorted_points[target_pt][0], sorted_points[target_pt][1]
            address = (f_ap, f_tp, round((t_tp - t_ap) * time_res, precision))
            couple = (round(t_ap * time_res, precision), songId)

            if address not in pairs.keys():
                pairs[address] = []
            pairs[address].append(couple if insert else couple[0])

    if insert:
        for address, couples in pairs.items():
            address = np.array(address).tolist()
            if len(list(db.pairs.find({"address": address}))) == 0:
                db.pairs.insert_one({"address": address, "couples": couples})
            else:
                db.pairs.find_one_and_update(
                    {"address": address}, {"$addToSet": {"couples": {"$each": couples}}}
                )
    return pairs, total_pts


# %%
def filter_database_song(candidate_song_ids):
    couples_grouped_by_song_id = {}
    for song_id in candidate_song_ids:
        couples_grouped_by_song_id[song_id] = {}
        pipeline = [
            {"$unwind": "$couples"},
            {"$match": {"couples.1": song_id}},
            {
                "$group": {
                    "_id": "$_id",
                    "address": {"$first": "$address"},
                    "t_ap": {"$push": {"$first": "$couples"}},
                }
            },
            {"$project": {"_id": 0}},
        ]
        song_pairs = list(db.pairs.aggregate(pipeline))
        for pair in song_pairs:
            address = tuple(pair["address"])
            t_ap = pair["t_ap"]
            if address in couples_grouped_by_song_id[song_id]:
                couples_grouped_by_song_id[song_id][address].extend(t_ap)
            else:
                couples_grouped_by_song_id[song_id][address] = t_ap
    return couples_grouped_by_song_id


# %%
# build fingerprint database
if build_fingerprint_db:
    audio_paths = glob.glob("./songs/*")
    song_id = 1

    for audio_path in audio_paths:
        print(song_id, audio_path)
        filtered_spectrogram = compute_spectrogram(audio_path)
        hash_song(filtered_spectrogram, song_id, insert=True)
        song_id += 1

# %%
test_audio_path = "test/test_5.wav"
filtered_spectrogram = compute_spectrogram(
    test_audio_path, plot_spec=True, plot_filtered_spec=True
)
pairs, total_pts = hash_song(filtered_spectrogram)

# %%
matches = []

for address in pairs.keys():
    address = np.array(address).tolist()
    if len(list(db.pairs.find({"address": address}))):
        f_ap = address[0]
        couples = db.pairs.find_one(
            {"address": address}, projection={"couples": 1, "_id": 0}
        )["couples"]
        matches.append((f_ap, [tuple(couple) for couple in couples]))

# result filtering

hits = {}
for (f_ap, couples) in matches:
    for couple in couples:
        if (f_ap, couple) not in hits:
            hits[(f_ap, couple)] = 0
        hits[(f_ap, couple)] += 1


filtered_matches = [match for match, count in hits.items() if count >= target_zone_size]

common_target_zones = {}
for match in filtered_matches:
    song_id = match[1][1]
    if song_id not in common_target_zones:
        common_target_zones[song_id] = 0
    common_target_zones[song_id] += 1

candidate_song_ids = []
for song_id, count in common_target_zones.items():
    if count >= total_pts * target_zone_coeff:
        candidate_song_ids.append(song_id)

if len(candidate_song_ids) == 0:
    print("No strong matches, finding best weak match...")
    candidate_song_ids = sorted(common_target_zones.keys())

# %%
# time coherency
couples_grouped_by_song_id = filter_database_song(candidate_song_ids)
coherent_notes = {}

best_match_song_id = -1
max_delta = 0

for song_id, pairs_dict in couples_grouped_by_song_id.items():
    deltas = {}
    for address, t_aps in pairs.items():
        for t_ap in t_aps:
            if address not in pairs_dict:
                continue
            t_tps = pairs_dict[address]
            for t_tp in t_tps:
                delta = round(t_tp - t_ap, precision)
                if delta not in deltas:
                    deltas[delta] = 0
                deltas[delta] += 1
    # keep best delta
    best_delta = max(deltas, key=deltas.get)
    max_delta = max(deltas[best_delta], max_delta)
    coherent_notes[song_id] = (best_delta, deltas[best_delta])
    best_match_song_id = (
        best_match_song_id if deltas[best_delta] < max_delta else song_id
    )

print(
    "Best match song id:",
    best_match_song_id,
    "\ndelta =",
    abs(coherent_notes[best_match_song_id][0]),
    "seconds",
)
print(
    json.dumps(db.info.find_one({"song_id": best_match_song_id}, {"_id": 0}), indent=4)
)
