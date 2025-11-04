# prepare_faq.py: Extract 100 FAQ entries from CSV and track with DVC
# Outputs: data/faq.json and tracks it with DVC

import pandas as pd
import json
import os
import random

# Step 1: Load and analyze CSV
csv_path = 'Customer_Service_Training.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} utterances. Unique intents: {df['intent'].nunique()}")

# Group by intent and sample ~4 queries per intent (target 100 total)
faq_entries = []
intent_groups = df.groupby('intent')
samples_per_intent = 4  # Adjust if needed; total ~100

answer_templates = { #27 intents
    'cancel_order': "To cancel your order, log in to your account, go to 'My Orders', select the order by {order_id}, and click 'Cancel'. If past deadline, contact support at support@company.com.",
    'change_order': "To change your order, visit 'My Orders' in your account. Edit items/quantity before shipment. For shipped orders, reply with {order_id} for assistance.",
    'change_shipping_address': "Update your shipping address in 'Account Settings' > 'Addresses'. Save changes and confirm for future orders. Need help? Reply with details.",
    'check_cancellation_fee': "Cancellation fees vary by order: 10% within 24h, full after. Check details in 'Order History' or reply with {order_id} for exact amount.",
    'check_invoice': "View invoices in 'Account' > 'Billing & Invoices'. Download PDF for specific dates/orders. If missing, provide {order_id} for resend.",
    'check_payment_methods': "We accept Visa, Mastercard, PayPal, and Apple Pay. See full list in 'Checkout' > 'Payment Options'. Questions? Reply here.",
    'check_refund_policy': "Refunds processed within 7-10 business days for eligible items. Full policy at help.company.com/refunds. Track status with {order_id}.",
    'complaint': "We're sorry for the issue. File a complaint at support@company.com with details/{order_id}. We'll respond within 48h. How can we help now?",
    'contact_customer_service': "Customer service hours: Mon-Fri 9AM-6PM EST. Call 1-800-123-4567 or chat via app. Reply for immediate guidance.",
    'contact_human_agent': "I'll connect you to a live agent shortly. Estimated wait: 2-5 min. In meantime, share your issue for faster resolution.",
    'create_account': "To create an account, visit company.com/signup. Enter email, password, and verify. Need assistance? Reply with questions.",
    'delete_account': "To delete your account, go to 'Account Settings' > 'Privacy' > 'Delete Account'. Confirm via email. Data retained 30 days.",
    'delivery_options': "Delivery options: Standard (3-5 days, free), Express (1-2 days, $5). Select at checkout. Track via 'My Orders'.",
    'delivery_period': "Standard delivery: 3-5 business days. Track real-time status in 'My Orders'. Delays? Reply with {order_id}.",
    'edit_account': "Edit account info in 'Account Settings' > 'Profile'. Update name/email/address and save. Verify changes via email.",
    'get_invoice': "Invoices available in 'Billing' section. Reply with {order_id}/date for direct send to your email.",
    'get_refund': "To request refund, go to 'My Orders' > 'Return/Refund'. Submit reason and track status. Processed in 5-7 days.",
    'newsletter_subscription': "Subscribe/unsubscribe to newsletter in 'Account' > 'Preferences' > 'Emails'. Confirm to start receiving updates.",
    'payment_issue': "Payment issues? Check card details or try alternate method. Reply with error message/{order_id} for troubleshooting.",
    'place_order': "To place an order, browse products, add to cart, and checkout. New customer? Create account first for faster process.",
    'recover_password': "Reset password at company.com/forgot-password. Enter email, follow link. If issues, reply for manual reset.",
    'registration_problems': "Registration error? Clear cache or try incognito mode. Still stuck? Reply with error screenshot/{details} for help.",
    'review': "Leave a review in 'Order Details' > 'Rate & Review'. Your feedback helps us improve—thank you!",
    'set_up_shipping_address': "Set up shipping address in 'Account' > 'Addresses' > 'Add New'. Enter details and set as default. Verify for accuracy.",
    'switch_account': "Switch accounts via top-right profile icon > 'Switch User'. Log in with credentials. Need multi-account help? Reply here.",
    'track_order': "Track order in 'My Orders' with tracking ID. Or reply here with {order_id} for updates.",
    'track_refund': "Refund status in 'Billing' > 'Refunds'. Reply with transaction ID for latest update."
}

for intent, group in intent_groups:
    queries = group['utterance'].dropna().tolist()  # Drop NaN
    sampled_queries = random.sample(queries, min(samples_per_intent, len(queries)))  # Sample up to 4
    for query in sampled_queries:
        answer = answer_templates.get(intent, "For this query, please visit our help center or reply for personalized assistance.")  # Fallback
        faq_entries.append({"query": query.strip(), "answer": answer})

# Shuffle to mix intents
random.shuffle(faq_entries)
faq_entries = faq_entries[:108]  # Ensure exactly 108

# Step 2: Save to JSON
os.makedirs('data', exist_ok=True)
json_path = 'data/faq.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(faq_entries, f, indent=2, ensure_ascii=False)
print(f"Saved {len(faq_entries)} FAQ entries to {json_path}")

