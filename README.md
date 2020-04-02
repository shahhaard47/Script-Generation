## **Pytorch-Genre-Based-Script-Generation**

## Data

Download genre.csv file and store it in ./data/


## Fine-Tune on Movie Script dataset

```
python train_scripts.py
```
Note: Edit argument parameters if you want to use non-default hyper-parameters. 


## Generate Style-based scripts using trained model

1. The following command fine-tunes the chosen model on SWAG. The available models are GPT-2. (BERT and RoBERTa should work but I haven't tested them yet)
```
python generate_scripts.py \
--text "text prompt to initialize generation" \
--genres "<Genre1> <Genre2> <Genre3> ....." \
--checkpoint './models/<checkpoint_folder>'
--learning_rate 3e-4 
```

Note: For initializing genres make sure you follow the same format as shown above, ex: "<Comedy> <Action>"

## License

- OpenAi/GPT2 follow MIT license, huggingface/pytorch-pretrained-BERT is Apache license. 
- I follow MIT license with original GPT2 repository

