import pandas as pd

import respiratory_sounds


def get_num_patients_per_diagnosis_class(relative=False):
    return patients['diagnosis_class'].value_counts(normalize=relative)


def get_num_recordings_per_diagnosis_class(relative=False):
    return recordings_patients_join['diagnosis_class'].value_counts(normalize=relative)


def get_mean_age_of_patients_in_recordings_grouped_by_class():
    recordings_patients_join['age'] = pd.to_numeric(recordings_patients_join['age'], errors='coerce')
    recordings_patients_join_no_missing_age = recordings_patients_join.dropna(subset=['age'])
    return recordings_patients_join_no_missing_age.groupby(['diagnosis_class'])[['age']].mean()


def get_mean_age_of_patients_grouped_by_class():
    patients['age'] = pd.to_numeric(patients['age'], errors='coerce')
    patients_no_missing_age = patients.dropna(subset=['age'])
    return patients_no_missing_age.groupby(['diagnosis_class'])[['age']].mean()


def get_num_recordings_per_patient(relative=False):
    return recordings_patients_join['patient_number'].value_counts(normalize=relative)


def get_distribution_of_recording_devices_grouped_by_class():
    recording_equipment, diagnosis_class = 'recording_equipment', 'diagnosis_class'
    samples_no_missing_vals = recordings_patients_join.dropna(subset=[recording_equipment])
    return samples_no_missing_vals.groupby([diagnosis_class, recording_equipment]).size()


def get_mean_variance_max_recording_length():
    recording_lengths = recordings_patients_join.apply(lambda row: len(row.audio_recording) / row.sample_rate, axis=1)
    return recording_lengths.mean(), recording_lengths.var(), recording_lengths.max()


def main():
    num_recordings_per_diagnosis_class = get_num_recordings_per_diagnosis_class()
    relative_num_recordings_per_diagnosis_class = get_num_recordings_per_diagnosis_class(relative=True)

    num_patients_per_diagnosis_class = get_num_patients_per_diagnosis_class()
    relative_num_patients_per_diagnosis_class = get_num_patients_per_diagnosis_class(relative=True)

    mean_age_patients_recordings = get_mean_age_of_patients_in_recordings_grouped_by_class()
    mean_age_patients = get_mean_age_of_patients_grouped_by_class()
    num_recordings_per_patient = get_num_recordings_per_patient()

    print("num recordings: ", num_recordings)
    print(len(recordings.index))
    for k, v in relative_num_recordings_per_diagnosis_class.items():
        print("{} -> {:.2f}% ({})".format(respiratory_sounds.convert_diagnosis_class_to_name(k), v * 100,
                                          num_recordings_per_diagnosis_class[k]))

    print(mean_age_patients_recordings)
    print(mean_age_patients)
    print(num_recordings_per_patient.value_counts().to_string())

    print(get_distribution_of_recording_devices_grouped_by_class())

    print(recordings_patients_join['sample_rate'].value_counts())

    print(get_mean_variance_max_recording_length())


recordings = respiratory_sounds.load_recordings()
patients = respiratory_sounds.load_patients()
recordings_patients_join = respiratory_sounds.get_recordings_patients_join()

num_recordings = len(recordings.index)
num_patients = len(patients.index)

if __name__ == '__main__':
    main()
