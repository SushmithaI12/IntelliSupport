"""Small FAQ knowledge base with keyword retrieval, plus a mock order database."""
import re

FAQS = [
    {"q": "How do I get a refund?",
     "a": "Refunds are available within 30 days of delivery for unused items. Go to My Orders, choose the item and click Request Refund. Money returns to the original payment method in 5 to 7 business days.",
     "keywords": ["refund", "money back", "return", "returns"]},
    {"q": "How long does shipping take?",
     "a": "Standard shipping takes 3 to 5 business days and express shipping takes 1 to 2 business days. Orders placed after 2 PM ship the next business day.",
     "keywords": ["shipping", "delivery", "deliver", "arrive", "how long"]},
    {"q": "How do I reset my password?",
     "a": "Click Forgot Password on the login page, enter your registered email and open the reset link we send. The link is valid for 30 minutes.",
     "keywords": ["password", "reset", "forgot", "login", "log in", "sign in"]},
    {"q": "Why was I charged twice?",
     "a": "A duplicate charge is usually a temporary authorization hold that disappears in 3 to 5 business days. If a second charge is still there after 5 days, share your order ID so we can investigate.",
     "keywords": ["charged twice", "double charge", "duplicate", "charged", "billing"]},
    {"q": "How do I update my payment method?",
     "a": "Open Account Settings, then Payment Methods. Add a new card and set it as default. Saved cards can be removed at any time.",
     "keywords": ["payment method", "card", "update card", "payment"]},
    {"q": "How can I track my order?",
     "a": "Use the order ID from your confirmation email in Track Order, or share the order ID with us and we will look it up.",
     "keywords": ["track", "tracking", "where is my order", "order status"]},
    {"q": "How do I cancel my order?",
     "a": "Orders can be cancelled before they ship. Open My Orders and click Cancel Order. Once an order has shipped it must be returned instead.",
     "keywords": ["cancel", "cancellation"]},
    {"q": "How do I change my delivery address?",
     "a": "The address can be changed before the order ships from My Orders, then Edit Address. After shipping, contact the courier with your tracking number.",
     "keywords": ["address", "change address", "wrong address"]},
    {"q": "The app keeps crashing. What should I do?",
     "a": "Update the app to the latest version, restart your device, and clear the app cache. If it still crashes, reinstall the app and log in again.",
     "keywords": ["crash", "crashing", "app", "bug", "not working", "error", "freeze"]},
    {"q": "Do you offer a warranty?",
     "a": "All electronics include a 12-month limited warranty that covers manufacturing defects but not accidental damage.",
     "keywords": ["warranty", "guarantee", "defect", "broken"]},
    {"q": "How do I contact support?",
     "a": "This chat is available 24/7. For account-specific issues a human agent can reach you by email within 24 hours.",
     "keywords": ["contact", "support", "phone", "email us"]},
    {"q": "Do you ship internationally?",
     "a": "Yes, we ship to over 40 countries. International delivery takes 7 to 14 business days and customs fees may apply.",
     "keywords": ["international", "abroad", "overseas", "country", "customs"]},
]

ORDERS = {
    "ORD-1001": {"status": "Shipped", "carrier": "BlueDart", "eta": "2026-11-08"},
    "ORD-1002": {"status": "Delivered", "carrier": "Delhivery", "eta": "2026-11-01"},
    "ORD-1003": {"status": "Processing", "carrier": "Not assigned yet", "eta": "2026-11-12"},
}

_STOP = {"the", "and", "for", "are", "but", "you", "with", "this", "that", "have",
         "was", "what", "how", "can", "does", "from", "your", "our", "its", "my"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def search_kb(query: str, top_k: int = 2, min_score: int = 2) -> list:
    """Score each FAQ: keyword phrase hit = 2 points, shared word with the question = 1."""
    q_low, q_tokens = query.lower(), _tokens(query)
    scored = []
    for faq in FAQS:
        score = sum(2 for k in faq["keywords"] if k in q_low)
        score += len(q_tokens & _tokens(faq["q"]))
        if score >= min_score:
            scored.append((score, faq))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_k]]


def get_order_status(order_id: str) -> dict:
    """Mock tool: look up an order. Returns None when the order does not exist."""
    return ORDERS.get(order_id.upper())
