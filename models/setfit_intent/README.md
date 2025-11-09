---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: I have to switch to another account
- text: help regarding payment
- text: where can I get my invoice?
- text: I need assistance buying some items
- text: help me submitting some feedback for your company
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 27 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                    | Examples                                                                                                                                                                                                        |
|:-------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| edit_account             | <ul><li>'i have got to modify the suer data how do i do it'</li><li>'assistance to change the details included on my user profile'</li><li>'could you help me editing my account?'</li></ul>                    |
| track_order              | <ul><li>'check ETA of order'</li><li>"I don't know how I could check the status of my order"</li><li>'how to check the status of my order?'</li></ul>                                                           |
| delete_account           | <ul><li>'help me delete my account'</li><li>'can you tell me more about a user account deletion?'</li><li>'can you help me deleting the account?'</li></ul>                                                     |
| contact_human_agent      | <ul><li>'need assistance talking to an operator'</li><li>'want assistance talking with a person'</li><li>'assistance to contact a person'</li></ul>                                                             |
| review                   | <ul><li>'I have got to submit some feedback for your services'</li><li>'I want assistance leaving a review for a service'</li><li>'I need to submit some feedback'</li></ul>                                    |
| complaint                | <ul><li>'I have to file a complaint'</li><li>'is it possible to file a consumer claim'</li><li>"I don't know what to do to file a complaint"</li></ul>                                                          |
| track_refund             | <ul><li>'I have got to check the refund status'</li><li>'what do I have to do to track the refund?'</li><li>'I am trying to check if there is any updates on my refund'</li></ul>                               |
| change_order             | <ul><li>'can you help me add several items to an order?'</li><li>'problem with removing something from an order'</li><li>'information about changing orders'</li></ul>                                          |
| check_payment_methods    | <ul><li>'help to check your allowed payment methods'</li><li>'check what payment options are allowed'</li><li>'what do I have to do to list the fucking payment methods?'</li></ul>                             |
| get_refund               | <ul><li>'assistance to get refunds of money'</li><li>'how can I get refunds?'</li><li>'I need help getting my money back'</li></ul>                                                                             |
| switch_account           | <ul><li>'how can I switch to another account?'</li><li>"I don't know how to switch user"</li><li>'switch to another account'</li></ul>                                                                          |
| recover_password         | <ul><li>'give me information about forgotten passwords'</li><li>'I cannot remember my key, how can I recover it?'</li><li>'help recovering my password'</li></ul>                                               |
| check_invoice            | <ul><li>'can you help me take a quick look at my invoice?'</li><li>'I do not know what I need to do to find the invoice'</li><li>'can you help me check my invoice?'</li></ul>                                  |
| check_refund_policy      | <ul><li>'what is your money back policy?'</li><li>'can I check how long refunds take?'</li><li>'checking your money back policy'</li></ul>                                                                      |
| place_order              | <ul><li>'how could I make a purchase?'</li><li>'I do not know how I could buy some items'</li><li>'making purchase'</li></ul>                                                                                   |
| delivery_options         | <ul><li>'I am looking for the options for delivery'</li><li>'help me check the delivery options'</li><li>'help me to check what delivery options you offer'</li></ul>                                           |
| payment_issue            | <ul><li>"I don't know what I need to do to report payment issues"</li><li>'can you help me report an issue with payments?'</li><li>"I don't know how to report payment issues"</li></ul>                        |
| cancel_order             | <ul><li>'I need help to cancel the order I made'</li><li>'how do I cancel an order I made?'</li><li>'help with canceling the order I made'</li></ul>                                                            |
| create_account           | <ul><li>'help me create an account'</li><li>'i need assistance to open a user account'</li><li>'want help creating another freemium account'</li></ul>                                                          |
| contact_customer_service | <ul><li>'what hours is customer service available?'</li><li>'will you show me what hours customer service available is?'</li><li>'I want help to speak to customer service'</li></ul>                           |
| set_up_shipping_address  | <ul><li>'issues setting up a new shipping address'</li><li>"why isn't my new shipping address valid"</li><li>'how could I set up a different shipping address?'</li></ul>                                       |
| delivery_period          | <ul><li>'help me to check when my delivery is going to arrive'</li><li>'can you help me checking when my item is going to arrive?'</li><li>'help checking when will my item arrive'</li></ul>                   |
| registration_problems    | <ul><li>"I can't report a registration issue"</li><li>'I cannot report registration issues'</li><li>'I have to report a registration issue'</li></ul>                                                           |
| check_cancellation_fee   | <ul><li>'want assistance seeing the withdrawal fee'</li><li>'could you help me to check the cancellation fees?'</li><li>'I do not know what I have to do to see the cancellatino fees'</li></ul>                |
| newsletter_subscription  | <ul><li>'where could i receive ur company newsletter'</li><li>'can you help me cancel my corporate newsletter subscription?'</li><li>'I have got to cancel my corporate newsletter subscription'</li></ul>      |
| get_invoice              | <ul><li>'where can I get the invoices from my last purchase?'</li><li>'I do not know how I can get the invoices from ten months ago'</li><li>'could you help me to get the invoices from last month?'</li></ul> |
| change_shipping_address  | <ul><li>'my shipping address has changed'</li><li>'I would like information about a shipping address change'</li><li>'i do not know how i can edit my address'</li></ul>                                        |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("help regarding payment")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 1   | 8.3221 | 15  |

| Label                    | Training Sample Count |
|:-------------------------|:----------------------|
| cancel_order             | 197                   |
| change_order             | 186                   |
| change_shipping_address  | 183                   |
| check_cancellation_fee   | 197                   |
| check_invoice            | 206                   |
| check_payment_methods    | 197                   |
| check_refund_policy      | 192                   |
| complaint                | 202                   |
| contact_customer_service | 202                   |
| contact_human_agent      | 180                   |
| create_account           | 190                   |
| delete_account           | 189                   |
| delivery_options         | 189                   |
| delivery_period          | 189                   |
| edit_account             | 186                   |
| get_invoice              | 214                   |
| get_refund               | 191                   |
| newsletter_subscription  | 190                   |
| payment_issue            | 210                   |
| place_order              | 194                   |
| recover_password         | 191                   |
| registration_problems    | 191                   |
| review                   | 198                   |
| set_up_shipping_address  | 197                   |
| switch_account           | 183                   |
| track_order              | 196                   |
| track_refund             | 191                   |

### Training Hyperparameters
- batch_size: (32, 32)
- num_epochs: (0, 0)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 1
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Framework Versions
- Python: 3.13.1
- SetFit: 1.1.3
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0+cpu
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->