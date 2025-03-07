# Popular models

## Models and their popular usage

| **Model Name**         | **Identifier on Hugging Face**      | **Primary Use Case for Sales Transactions**                  | **Why Popular**                                      | **Popularity Level** | **Key Strength**                  |
|-------------------------|-------------------------------------|-------------------------------------------------------------|-----------------------------------------------------|----------------------|-----------------------------------|
| **BERT**               | `bert-base-uncased`                | Text classification, entity recognition, sentiment analysis | Foundational NLP model with bidirectional context   | Extremely High       | Versatile, widely supported       |
| **DistilBERT**         | `distilbert-base-uncased`          | Lightweight categorization or sentiment analysis            | 97% of BERT’s performance, 40% fewer parameters     | Very High            | Efficient and fast                |
| **RoBERTa**            | `roberta-base`                     | Advanced classification of transaction types, feedback      | Outperforms BERT with optimized training            | High                 | High performance on benchmarks    |
| **T5**                 | `t5-base`                          | Generating summaries or structured outputs from data        | Flexible text-to-text framework                     | Moderately High      | Adaptable to multiple tasks       |
| **FinBERT**            | `ProsusAI/finbert`                 | Financial sentiment, transaction analysis                   | Pre-trained on financial texts                      | Moderate             | Domain-specific (finance/sales)   |
| **mBART**              | `facebook/mbart-large-50`          | Multilingual transaction processing                         | Supports 50 languages                               | Moderate             | Multilingual capabilities         |
| **ALBERT**             | `albert-base-v2`                   | Lightweight categorization or clustering                   | Reduced memory usage with decent performance        | Moderate             | Resource-efficient                |
| **MarianMT**           | `Helsinki-NLP/opus-mt-en-de` (varies by pair) | Translation of transaction descriptions                     | Specialized for translation across language pairs   | Lower                | Effective for specific languages  |

### Notes:
- **Popularity Level**: Estimated based on community adoption, downloads, and general-purpose usage.
- **Key Strength**: Highlights the standout feature for sales transaction tasks.
- **Customization**: All models require fine-tuning on a sales-specific dataset (e.g., transaction logs, customer notes).
- **Access**: Available on [Hugging Face’s Model Hub](https://huggingface.co/models) using the listed identifiers.
