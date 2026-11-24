import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from forgery.sensitivity import cluster_labels, hamming_matrix, leakage_table


def test_hamming_and_clusters():
    hashes = ["0000", "0001", "1111", "1110"]
    distance = hamming_matrix(hashes)
    assert distance[0, 1] == 1 and distance[0, 2] == 4 and distance[2, 3] == 1
    labels = cluster_labels(distance, 1)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


def test_leakage_counts_held_out_images_with_a_training_twin():
    hashes = ["0000", "0001", "1111", "1110", "0110"]
    frame = pd.DataFrame({"random_split": ["train", "val", "val", "train", "val"]})
    table = leakage_table(hamming_matrix(hashes), frame)
    row = table[table.hamming_threshold == 8].iloc[0]
    assert row.random_split_held_out == 3
    tight = table.iloc[0]
    assert tight.random_split_leaked <= row.random_split_leaked
