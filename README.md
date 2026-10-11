# WheezeDetect — Respiratory Sound Analysis

## Project Overview

Respiratory audio contains acoustic patterns that can be studied using
digital signal processing and machine learning. Wheeze-like sounds may
be difficult to identify automatically because recordings can differ in
duration, intensity, background noise, and recording conditions.

WheezeDetect explores a computational workflow for analysing prerecorded
respiratory audio. The training pipeline extracts acoustic features from
overlapping windows and trains a machine-learning classifier. The saved
model is then loaded by a Flask backend and connected to a web
interface.

### Project goals

-   Process respiratory audio in a consistent format.
-   Divide recordings into overlapping segments for local analysis.
-   Extract MFCC, temporal-derivative, and spectral features.
-   Apply a trained Extra Trees classifier to segment-level features.
-   Present model outputs and waveform information through a browser
    interface.
-   Evaluate performance using standard classification metrics and
    document limitations transparently.

## Features

-   **Audio upload and playback** --- select and listen to supported WAV
    recordings.
-   **Waveform visualisation** --- inspect amplitude changes over time.
-   **Segment-level scoring** --- view individual window scores when the
    backend returns them.
-   **Prediction summary** --- display the model output, score,
    threshold, duration, and number of analysed segments.
-   **Analysis dashboard** --- view summary information for analyses
    saved in the browser.
-   **History and CSV export** --- review locally stored analysis
    records and export them where implemented.
-   **Backend health check** --- check whether the Flask API is
    reachable and the model is loaded.
-   **Demo mode** --- test interface behaviour with simulated output,
    clearly distinguished from real model inference.

> The waveform is a visual representation of the audio signal; it is not
> itself the model's prediction. Demo-mode results are simulated and
> must never be interpreted as actual inference.

## How It Works

``` text
Respiratory WAV Recording
          |
          v
Audio Loading and Validation
          |
          v
Mono Conversion and Resampling to 16 kHz
          |
          v
2-Second Windows with 1-Second Stride
          |
          v
Acoustic Feature Extraction
(MFCC + Delta + Delta-Delta + Spectral Features)
          |
          v
Trained Extra Trees Classifier
          |
          v
Segment Scores
          |
          v
Top-1 Aggregation and Decision Threshold
          |
          v
Flask API Response
          |
          v
WheezeDetect Web Interface
```

The model output indicates whether the recording meets the configured
model threshold for a **wheeze-like pattern**. It does not establish the
presence or absence of a medical condition.

## Machine-Learning Pipeline

### Dataset

The project uses the Respiratory Sound Database obtained through Kaggle,
based on the ICBHI respiratory sound dataset. The working dataset used
during development contained **920 recordings from 126 patients**.

Dataset versions, file availability, annotation coverage, and
preprocessing choices can affect the final counts. The training pipeline
uses patient-grouped splitting so that a patient's recordings are not
intentionally shared across the training and test groups.

### Audio preprocessing

  -----------------------------------------------------------------------
  Setting                             Configuration
  ----------------------------------- -----------------------------------
  Target sample rate                  16,000 Hz

  Audio channels                      Mono

  Window duration                     2 seconds

  Window stride                       1 second

  FFT size                            512

  Hop length                          160 samples

  MFCC coefficients                   40

  Segment aggregation                 `top_1` / maximum segment score

  Decision threshold                  0.225 in the saved experiment
                                      configuration; verify against the
                                      loaded model artifact
  -----------------------------------------------------------------------

### Acoustic feature extraction

The feature extractor calculates summary statistics for multiple feature
families:

-   **MFCC mean and standard deviation:** describe the segment's
    spectral envelope.
-   **Delta MFCC statistics:** describe changes in MFCC values over
    time.
-   **Delta-delta MFCC statistics:** describe changes in the delta
    features.
-   **Spectral centroid:** represents the centre of spectral energy.
-   **Spectral bandwidth:** describes spectral spread.
-   **Spectral rolloff:** summarises the frequency distribution below a
    selected energy proportion.
-   **Zero-crossing rate:** measures how often the waveform changes
    sign.
-   **RMS energy:** summarises signal energy.
-   **Spectral contrast:** describes differences between spectral peaks
    and valleys across frequency bands.

The trained artifact includes the classifier and inference
configuration, such as feature-column names, sample rate, window
duration, stride, aggregation method, and threshold. The backend must
use the same feature definitions and ordering as training.

### Model

The current model is an **Extra Trees Classifier**. The training
experiment used patient-grouped evaluation and selected its model
configuration using training-only out-of-fold predictions. The final
held-out test set was used to report evaluation metrics.

## Model Evaluation

The current recorded held-out evaluation contains **160 recordings**.

  Metric              Result
  ----------------- --------
  Accuracy            57.50%
  Precision           55.07%
  Recall              92.68%
  F1-score            69.09%
  True negatives          16
  False positives         62
  False negatives          6
  True positives          76

These results show high recall but limited precision. The model
identifies many recordings labelled as wheeze, but it also produces many
false-positive predictions. Accuracy and specificity therefore remain
important limitations.

The reported values describe the recorded experiment and are not a
guarantee of performance on other recordings, devices, populations, or
environments. A model score is not necessarily a calibrated medical
probability.

## Technology Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Audio processing and model
                                      development

  Librosa                             Audio loading, resampling, and
                                      feature extraction

  NumPy                               Numerical signal operations

  Pandas                              Dataset, features, and results
                                      management

  scikit-learn                        Extra Trees classifier and
                                      evaluation metrics

  Joblib                              Saving and loading the trained
                                      model

  Flask                               Backend API and inference

  HTML                                Frontend structure

  CSS                                 Responsive interface styling

  JavaScript                          Upload interactions, API
                                      communication, and visualisation

  Google Colab                        Model training and experimentation

  Kaggle                              Respiratory sound dataset source
  -----------------------------------------------------------------------

## Project Structure

The expected structure is:

``` text
WheezeDetect/
├── index.html
├── style.css
├── script.js
├── logo.svg
├── lucide.min.js
├── README.md
└── backend/
    ├── app.py
    ├── model_utils.py
    ├── test_backend.py
    ├── requirements.txt
    └── models/
        └── improved_wheeze_model.joblib
```

Some files may differ depending on your current repository version. Keep
the model artifact at the path expected by `backend/model_utils.py`, or
update the configured model path accordingly.

## Getting Started

### Prerequisites

-   Python 3.10 or newer, preferably a version supported by the
    dependencies in `backend/requirements.txt`.
-   Git, if cloning the repository.
-   A modern browser such as Chrome, Firefox, Edge, or Safari.
-   A supported WAV audio file for testing.
-   The trained model artifact in `backend/models/`.

### 1. Clone the repository

Replace the placeholder URL with your actual GitHub repository URL.

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd WheezeDetect
```

Alternatively, download the repository ZIP and extract it.

### 2. Create a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

``` bat
.venv\Scripts\activate.bat
```

**macOS/Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

From the project root:

``` bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

If installation fails, check that your Python version is compatible with
the packages listed in `backend/requirements.txt`.

### 4. Check the model file

Confirm that the following file exists:

``` text
backend/models/improved_wheeze_model.joblib
```

The model file is required for real inference. If it is not included in
the repository, obtain it from the project's authorised storage and
place it at the expected path. Do not commit large model files or
dataset recordings without checking repository limits, licensing, and
redistribution permissions.

## Running WheezeDetect

From the project root, run:

``` bash
python backend/app.py
```

Read the terminal output for the actual host and port. If the
application starts on the default local Flask address, open:

``` text
http://127.0.0.1:5000/
```

If the frontend is not served by Flask in your current version, open
`index.html` in a browser or use a local static server. For real
predictions, the frontend must be configured to send requests to the
running Flask API. Opening the frontend alone does not start the
backend.

### Verify backend health

Open this endpoint in your browser:

``` text
http://127.0.0.1:5000/api/health
```

A healthy response should indicate that the service is running and
report whether the model loaded successfully. Confirm the actual JSON
field names in your implementation.

### Analyse an audio file

1.  Start the Flask backend.
2.  Open the WheezeDetect frontend.
3.  Select a supported WAV file.
4.  Confirm that the correct file is displayed and can be played.
5.  Ensure **Demo Mode** is off when testing real inference.
6.  Select **Analyse Audio**.
7.  Review the returned result, score, threshold, duration, and segment
    information.
8.  Check the backend terminal for errors if the request fails.

Use the same test audio files when comparing outputs from different
environments. A backend health check confirms connectivity and model
loading; it does **not** prove that preprocessing and predictions match
the training pipeline.

## API Reference

The following endpoints describe the expected API contract in this
project. Confirm the exact response fields against `backend/app.py`,
because implementations can differ.

### `GET /api/health`

Checks backend availability and model-loading status.

Example response:

``` json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Extra Trees Classifier",
  "sample_rate": 16000,
  "threshold": 0.225
}
```

This is an illustrative response shape; the actual fields depend on the
current backend implementation.

### `POST /api/analyze`

Uploads and analyses an audio recording.

-   **Content type:** `multipart/form-data`
-   **File field:** `file`
-   **Expected format:** WAV
-   **Maximum file size:** 50 MB if configured in the backend
-   **Minimum duration:** 2 seconds for a full analysis window

Example request using `curl`:

``` bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -F "file=@sample_audio.wav"
```

An analysis response may include the filename, prediction label, model
score, threshold, duration, number of segments, per-segment scores, and
model name. Use the actual response produced by your backend as the
source of truth.

## Testing

If `backend/test_backend.py` is present, run:

``` bash
python backend/test_backend.py
```

Check the test output to confirm which tests actually ran and passed.
Tests using synthetic audio or mocked inference can verify API behaviour
and input validation, but they do not prove that the trained model is
accurate or that the feature extractor exactly matches the training
pipeline.

For a stronger inference consistency check:

1.  Select a known WAV file.
2.  Calculate its SHA-256 hash in each environment to confirm the files
    are identical.
3.  Run the original training/inference pipeline and the Flask backend
    on that same file.
4.  Compare sample rate, segment count, feature-column order,
    per-segment scores, aggregation, threshold, and final prediction.
5.  Investigate any differences before describing the integration as
    verified.

## Troubleshooting

### Backend does not start

-   Activate the virtual environment.
-   Install dependencies from `backend/requirements.txt`.
-   Check the terminal traceback.
-   Confirm that the model path and required files exist.
-   Check whether port `5000` is already in use.

### Model not loaded

-   Confirm that `improved_wheeze_model.joblib` exists at the configured
    location.
-   Verify that the environment has compatible versions of the required
    libraries.
-   Check the model path used in `model_utils.py`.
-   Review the backend startup error rather than relying only on the
    frontend status indicator.

### Audio upload fails

-   Use a valid WAV file.
-   Check the configured file-size limit.
-   Ensure the recording is at least two seconds long.
-   Inspect the browser console and Flask terminal for the error
    message.

### Icons or styling do not update

-   Hard-refresh the page with `Ctrl + F5`.
-   Check that `style.css`, the logo asset, and any local icon library
    are loading correctly.
-   Confirm that asset paths are relative to the actual location of the
    HTML file.

### Backend result differs from the training result

-   Confirm that both environments use the exact same audio file.
-   Verify mono conversion and resampling to 16 kHz.
-   Verify two-second windows with one-second stride.
-   Compare the feature extraction code and feature-column order.
-   Confirm the positive class used for `predict_proba`.
-   Confirm top-1 aggregation and the threshold loaded from the model
    artifact.
-   Compare individual segment scores before comparing the final
    recording-level prediction.

Do not fix discrepancies by hardcoding expected results or changing the
threshold simply to force a match.

## Limitations and Responsible Use

-   This is a research prototype, not a clinically validated diagnostic
    device.
-   The current held-out results contain a high number of false
    positives.
-   The score should not be interpreted as a calibrated probability of
    disease.
-   Performance may change across datasets, recording equipment, patient
    populations, and recording environments.
-   Waveform visualisation describes the audio signal; it does not
    independently identify wheezing.
-   The interface's demo mode produces simulated output and must be
    clearly distinguished from real model inference.
-   Clinical deployment would require independent validation,
    appropriate oversight, and evidence of safety and effectiveness.

**Never use WheezeDetect as a substitute for assessment by a qualified
healthcare professional.**

## Future Improvements

-   Improve precision and specificity while maintaining clinically
    appropriate sensitivity.
-   Verify training and inference feature extraction with automated
    parity tests.
-   Evaluate the model on independent datasets and unseen recording
    conditions.
-   Assess probability calibration and decision-threshold selection on
    validation data.
-   Add reproducible experiment tracking and model-version metadata.
-   Improve error handling for corrupted, short, or unsupported audio
    files.
-   Add automated tests for file validation, feature extraction, API
    responses, and inference consistency.
-   Improve waveform and segment-score visualisations without implying
    that waveform shape alone indicates a diagnosis.

## Acknowledgements

-   The Respiratory Sound Database / ICBHI respiratory sound dataset and
    its contributors.
-   The open-source Python audio-processing and machine-learning
    ecosystem, including Librosa, NumPy, Pandas, scikit-learn, Joblib,
    and Flask.
-   Google Colab and Kaggle for supporting model experimentation and
    dataset access.

Please follow the dataset's licence and usage terms when downloading,
using, or redistributing the data.

## License

Add a `LICENSE` file before declaring this project open source. Until a
licence is included, reuse and redistribution permissions should not be
assumed.

