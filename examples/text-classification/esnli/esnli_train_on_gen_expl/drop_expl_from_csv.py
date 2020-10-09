# make the gold expl column from training csv (in esnli_train.csv format) contain only empty strings 
# so that we could pretrain the classifier using p+h+e, where the e is an empty string
# and then we will fine-tune using the generated explanations

import csv


def main():
    input_csv_path = "/data/rosa/HF-transformers-20/examples/EncoderDecoderModel" \
                     "/esnli/esnli_task_trained_model_and_results/esnli/esnli_train_results/" \
                     "eval_on_train_expl_text_in_esnli_train_format.csv"

    output_csv_path = './pretrain_on_phe_empty_expl.csv'

    with open(output_csv_path, mode='w') as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(input_csv_path) as f:
            reader = csv.reader(f)
            for (i, line) in enumerate(reader):
                if i > 0:
                    line[4] = "" # replace expl with empty string
                writer.writerow(line) 


if __name__=="__main__":
    main()