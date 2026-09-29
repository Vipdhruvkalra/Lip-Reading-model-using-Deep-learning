# LipNet: Lip Reading with Deep Learning

An end-to-end lip reading model built with TensorFlow/Keras. Given a short silent video of a person speaking, LipNet predicts the sentence being spoken using only the movement of the lips.

The model is a 3D-CNN + Bidirectional LSTM network trained with **CTC (Connectionist Temporal Classification) loss**, following the architecture from the paper [*LipNet: End-to-End Sentence-level Lipreading*](https://arxiv.org/abs/1611.01599).

## How it works

```
Video (75 frames)
   -> crop mouth region, grayscale, normalize
   -> 3x Conv3D + MaxPool3D   (spatio-temporal features)
   -> TimeDistributed Flatten
   -> 2x Bidirectional LSTM   (temporal modeling)
   -> Dense + Softmax         (character probabilities per frame)
   -> CTC decoding            (predicted sentence)
```

### Data preprocessing
- Each video is read with OpenCV and converted to grayscale.
- The mouth region is cropped to a fixed window (`[190:236, 80:220]`, i.e. 46 x 140 pixels).
- Frames are standardized (zero mean, unit variance).
- Each clip is 75 frames long.
- Transcripts come from `.align` files. Silence tokens (`sil`) are removed and words are converted to character indices.

### Vocabulary
Lowercase letters `a-z`, digits `1-9`, space, and the characters `' ? !`, plus one extra class for the CTC blank token.

### Model architecture

| Layer | Details |
|-------|---------|
| Conv3D + ReLU + MaxPool3D | 128 filters, 3x3x3 kernel, pool (1,2,2) |
| Conv3D + ReLU + MaxPool3D | 256 filters, 3x3x3 kernel, pool (1,2,2) |
| Conv3D + ReLU + MaxPool3D | 75 filters, 3x3x3 kernel, pool (1,2,2) |
| TimeDistributed(Flatten) | Flattens features per frame |
| Bidirectional LSTM + Dropout(0.5) | 128 units, orthogonal init |
| Bidirectional LSTM + Dropout(0.5) | 128 units, orthogonal init |
| Dense + Softmax | `vocab_size + 1` outputs |

Input shape: `(75, 46, 140, 1)`

### Training setup
- **Loss:** CTC loss
- **Optimizer:** Adam (learning rate `1e-4`)
- **Learning rate schedule:** constant for 30 epochs, then exponential decay
- **Epochs:** 100
- **Batch size:** 2
- **Split:** 450 batches for training, the rest for testing
- **Callbacks:** model checkpointing, LR scheduler, and a callback that prints sample predictions after each epoch

## Dataset

The project uses speaker **s1** from the [GRID audiovisual corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/). The notebook downloads a prepared copy automatically with `gdown`, and extracts it into a `data/` folder:

```
data/
├── s1/                 # .mpg video files
└── alignments/
    └── s1/             # .align transcript files
```

## Getting started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
Python 3.10 is recommended.

```bash
pip install opencv-python matplotlib imageio gdown tensorflow
```

### 3. Run the notebook
```bash
jupyter notebook LipNet.ipynb
```

The notebook is organized into sections:

0. Install and import dependencies
1. Build data loading functions (downloads the dataset)
2. Create the data pipeline
3. Design the neural network
4. Set up training options and train
5. Make predictions (downloads pretrained weights)

### Skip training: use pretrained weights
Training for 100 epochs is slow. Section 5 of the notebook downloads pretrained checkpoints into `models/` and loads them:

```python
model.load_weights("models/lipnet_pretrained.weights.h5")
```

You can then run predictions on the test set or on a single video.

## Making a prediction

```python
sample = load_data(tf.convert_to_tensor('./data/s1/bras9a.mpg'))
yhat = model.predict(tf.expand_dims(sample[0], axis=0))
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

print([tf.strings.reduce_join([num_to_char(w) for w in s]) for s in decoded])
```

## Repository structure

```
.
├── LipNet.ipynb        # Full pipeline: data, model, training, inference
├── README.md
├── data/               # Downloaded automatically (not tracked in git)
└── models/             # Checkpoints / pretrained weights (not tracked in git)
```

> `data/` and `models/` are large and are excluded from the repo via `.gitignore`. The notebook downloads them when you run it.

## Notes and troubleshooting

- **GPU:** The notebook enables memory growth if a GPU is available, but it also runs on CPU (slowly).
- **`ctc_decode` errors:** `input_length` must contain one entry per sample in the batch. Use `[75]` for a single video and `[75, 75]` for a batch of 2, or use `[yhat.shape[1]] * yhat.shape[0]` to handle any batch size.
- **File paths:** The notebook uses Windows-style paths in a few places (`.\\data\\s1\\...`). The `load_data` function handles both Windows and macOS/Linux paths.

## Limitations

- Trained on a single speaker (s1) from GRID, so it will not generalize well to other speakers, lighting, or camera angles.
- The mouth crop is hard-coded for this dataset's framing.
- GRID sentences follow a fixed grammar, so the model does not handle free-form speech.

## Acknowledgements

- [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599) (Assael et al., 2016)
- [GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- Built with TensorFlow, OpenCV, and imageio
