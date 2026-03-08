import argparse
import time

import numpy as np
import serial
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from feature_utils import (
    featureAR,
    featureMAV,
    featureMCV,
    featureME,
    featureMF,
    featureMPF,
    featureRMS,
    featureSM2,
    featureSSC,
    featureVAR,
    featureWA,
    featureWL,
    featureWPT,
    featureZC,
)


CLASS_ORDER = [("前", 0), ("后", 1), ("左", 2), ("右", 3)]


def parse_args():
    parser = argparse.ArgumentParser(description="EMG gesture recognition from serial stream.")
    parser.add_argument("--port", default="com6", help="Serial port, e.g. COM6 or /dev/tty.usbserial-*")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--rounds", type=int, default=3, help="How many front/back/left/right collection rounds")
    parser.add_argument("--samples-per-class", type=int, default=2000, help="Samples per class for one round")
    parser.add_argument("--window-size", type=int, default=200, help="Feature window size")
    return parser.parse_args()


def _parse_serial_line(raw_line):
    try:
        parts = raw_line.decode(errors="ignore").strip().split(",")
        values = np.array(list(map(float, parts)), dtype=float)
    except (ValueError, TypeError):
        return None

    if values.size < 3 or np.isnan(values).any():
        return None
    return values[:3]


def collect_samples(ser, target_count, class_name):
    samples = []
    print(class_name)
    time.sleep(1)

    while len(samples) < target_count:
        parsed = _parse_serial_line(ser.readline())
        if parsed is not None:
            samples.append(parsed)
    return np.array(samples, dtype=float)


def build_windows(class_samples, class_label, window_size):
    windows = []
    labels = []
    total = class_samples.shape[0] // window_size
    print("class", class_label, "number of sample:", class_samples.shape[0], total)
    for idx in range(total):
        start = idx * window_size
        end = start + window_size
        windows.append(class_samples[start:end, :])
        labels.append(class_label)
    return windows, labels


def extract_feature_vector(window_data):
    rms = featureRMS(window_data)
    mav = featureMAV(window_data)
    wl = featureWL(window_data)
    zc = featureZC(window_data)
    ssc = featureSSC(window_data)
    var = featureVAR(window_data)
    wa = featureWA(window_data)
    me = featureME(window_data)
    mcv = featureMCV(window_data)
    mfp = featureMPF(window_data)
    ar0, ar1, ar2, ar3, ar4, ar5, ar6 = featureAR(window_data)
    sm2 = featureSM2(window_data)
    mf = featureMF(window_data)
    e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5, v6, v7 = featureWPT(
        window_data
    )
    return np.hstack(
        (
            rms,
            mav,
            wl,
            zc,
            ssc,
            var,
            wa,
            me,
            mcv,
            mfp,
            ar0,
            ar1,
            ar2,
            ar3,
            ar4,
            ar5,
            ar6,
            sm2,
            mf,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            e7,
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
        )
    )


def make_feature_dataset(image_data, image_labels):
    feature_data = []
    feature_labels = []
    for idx in tqdm(range(image_data.shape[0])):
        feature_data.append(extract_feature_vector(image_data[idx]))
        feature_labels.append(image_labels[idx])
    return np.array(feature_data), np.array(feature_labels)


def main():
    args = parse_args()
    serial_conn = serial.Serial(args.port, args.baudrate)

    try:
        per_class_data = {label: [] for _, label in CLASS_ORDER}

        for _ in range(args.rounds):
            for class_name, class_id in CLASS_ORDER:
                samples = collect_samples(serial_conn, args.samples_per_class, class_name)
                per_class_data[class_id].append(samples)

        class_arrays = {}
        for _, class_id in CLASS_ORDER:
            class_arrays[class_id] = np.vstack(per_class_data[class_id])

        image_data = []
        image_label = []
        for _, class_id in CLASS_ORDER:
            windows, labels = build_windows(class_arrays[class_id], class_id, args.window_size)
            image_data.extend(windows)
            image_label.extend(labels)

        image_data = np.array(image_data)
        image_label = np.array(image_label)

        feature_data, feature_label = make_feature_dataset(image_data, image_label)

        train_x, test_x, train_y, test_y = train_test_split(
            feature_data, feature_label, test_size=0.2, random_state=42, stratify=feature_label
        )

        rf = RandomForestClassifier(
            n_estimators=180,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )

        rf.fit(train_x, train_y)
        train_score = rf.score(train_x, train_y)
        predict = rf.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, predict)

        print("RF train accuracy: %.2f%%" % (100 * train_score))
        print("RF test  accuracy: %.2f%%" % (100 * accuracy))

        while True:
            realtime_data = []
            while len(realtime_data) < args.window_size:
                parsed = _parse_serial_line(serial_conn.readline())
                if parsed is not None:
                    realtime_data.append(parsed)

            realtime_data = np.array(realtime_data, dtype=float)
            feature_vector = extract_feature_vector(realtime_data)
            prediction = rf.predict(np.array([feature_vector]))
            print(prediction)

    finally:
        serial_conn.close()


if __name__ == "__main__":
    main()
