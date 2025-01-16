from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from arabert.preprocess import ArabertPreprocessor
import json
import jsonlines
from transformers import pipeline


def summarize_text(file_path, output_file_path):
    MAX_INPUT_LENGTH = 10240  # Maximum length of the input to the model
    MIN_TARGET_LENGTH = 25  # Minimum length of the output by the model
    MAX_TARGET_LENGTH = 10240  # Maximum length of the output by the model

    model_name = "model_folder"
    preprocessor = ArabertPreprocessor(model_name="")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
    model.load_weights('AITC_TEAM.h5')

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

    json_list = []  # List to store the JSON objects
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            json_list.append(json_obj)

    # Define the column name for the new summary
    new_summary_column = 'summary'

    # Create a new list to store the updated JSON data
    updated_json_list = []

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")

    # Iterate over each text in json_list
    for i in range(len(json_list)):
        # Generate the summary for the current text
        summary = summarizer(
            json_list[i]['paragraph'],
            min_length=MIN_TARGET_LENGTH,
            max_length=MAX_TARGET_LENGTH,
        )[0]['summary_text']

        # Create a new dictionary to store the updated data
        del json_list[i]["paragraph"]
        updated_dict = dict(json_list[i])

        # Add the summary to the new dictionary
        updated_dict[new_summary_column] = summary

        # Append the updated dictionary to the new list
        updated_json_list.append(updated_dict)
        
    # Save the updated JSON data to a file in JSONL format
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for updated_dict in updated_json_list:
            updated_dict_str = json.dumps(updated_dict, ensure_ascii=False) + '\n'
            output_file.write(updated_dict_str)

    return f"Updated JSON data saved to: {output_file_path}"

input_file_path = 'validation_data.jsonl'
output_file_path = 'predictions.jsonl'

result = summarize_text(input_file_path, output_file_path)
print(result)