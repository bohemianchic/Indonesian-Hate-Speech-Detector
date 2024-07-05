# Indonesian-Hate-Speech-Detector

This project fine-tunes a pre-trained IndoBERTweet model for hate speech detection in Indonesian tweets. The model is quite effective. Emphasis on the 'quite'. This is particularly if your primary concern is not missing positive cases (high recall). 
Though, if your application requires minimising incorrect positive predictions (like in precision-sensitive contexts), you might consider methods to improve precision.  I had a problem looking for Indonesian dataset, I only used [this](https://github.com/ialfina/id-hatespeech-detection/blob/master/IDHSD_RIO_unbalanced_713_2017.txt) and [this](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection).

## Motivation

I am doing a project focusing on hate speech against Rohingya community in Aceh. I am not Indonesian nor do I speak the language. I have sought professional translator, don't worry. But I just thought I would fine-tune a model out of curiosity.


## Repository Structure
- `data/`: Contains the raw and processed datasets.
- `scripts/`: Contains the Python scripts for data loading, preparation, and model training.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and examples.
- `requirements.txt`: List of dependencies.

### Usage Example
For a detailed example, see the Jupyter notebook in the `notebooks/` directory:
- `notebooks/example_usage.ipynb`


## Model Performance Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.7742 |
| Precision | 0.7212 |
| Recall    | 0.8891 |
| F1-Score  | 0.7964 |


## Contributing
Feel free to submit issues or pull requests. Contributions are welcome!

