"""Fixed reference data for the synthetic ecosystem.

Bank, gateway and city names are labels only. Every number attached to them is synthetic and
says nothing about the real institutions.
"""

from dataclasses import dataclass

METHODS = ("UPI", "CARD", "NETBANKING")
DEVICES = ("android", "ios", "desktop")


@dataclass(frozen=True)
class City:
    name: str
    state: str
    weight: float


@dataclass(frozen=True)
class Issuer:
    name: str
    bank_type: str
    market_share: float


@dataclass(frozen=True)
class Gateway:
    name: str
    supported_methods: tuple[str, ...]
    routing_share: float


@dataclass(frozen=True)
class MerchantCategory:
    name: str
    weight: float
    average_order_value: float
    # Relative acceptance weight per method, aligned with METHODS.
    method_weights: tuple[float, float, float]
    # Online merchants draw customers from every city; offline ones mostly from their own.
    online: bool


CITIES = (
    City("Mumbai", "Maharashtra", 10.0),
    City("Delhi", "Delhi", 10.0),
    City("Bangalore", "Karnataka", 9.0),
    City("Hyderabad", "Telangana", 7.0),
    City("Chennai", "Tamil Nadu", 7.0),
    City("Kolkata", "West Bengal", 6.0),
    City("Pune", "Maharashtra", 5.0),
    City("Ahmedabad", "Gujarat", 4.5),
    City("Jaipur", "Rajasthan", 3.0),
    City("Lucknow", "Uttar Pradesh", 3.0),
    City("Surat", "Gujarat", 2.5),
    City("Kochi", "Kerala", 2.0),
    City("Chandigarh", "Chandigarh", 2.0),
    City("Indore", "Madhya Pradesh", 2.0),
    City("Bhopal", "Madhya Pradesh", 1.5),
    City("Nagpur", "Maharashtra", 1.5),
    City("Patna", "Bihar", 1.5),
    City("Coimbatore", "Tamil Nadu", 1.5),
    City("Visakhapatnam", "Andhra Pradesh", 1.5),
    City("Guwahati", "Assam", 1.0),
)

ISSUERS = (
    Issuer("HDFC", "private", 0.17),
    Issuer("SBI", "public", 0.20),
    Issuer("ICICI", "private", 0.14),
    Issuer("Axis", "private", 0.09),
    Issuer("Kotak", "private", 0.07),
    Issuer("PNB", "public", 0.06),
    Issuer("Bank of Baroda", "public", 0.06),
    Issuer("Canara", "public", 0.05),
    Issuer("Yes Bank", "private", 0.04),
    Issuer("IDFC First", "private", 0.04),
    Issuer("IndusInd", "private", 0.04),
    Issuer("AU Small Finance", "small_finance", 0.04),
)

GATEWAYS = (
    Gateway("gateway_a", ("UPI", "CARD", "NETBANKING"), 0.40),
    Gateway("gateway_b", ("UPI", "CARD", "NETBANKING"), 0.30),
    Gateway("gateway_c", ("UPI", "CARD"), 0.20),
    Gateway("gateway_d", ("UPI",), 0.10),
)

MERCHANT_CATEGORIES = (
    MerchantCategory("ecommerce", 0.18, 1400.0, (0.50, 0.35, 0.15), True),
    MerchantCategory("food_delivery", 0.16, 450.0, (0.75, 0.20, 0.05), True),
    MerchantCategory("travel", 0.07, 5200.0, (0.35, 0.45, 0.20), True),
    MerchantCategory("grocery", 0.14, 700.0, (0.80, 0.15, 0.05), False),
    MerchantCategory("utilities", 0.08, 1800.0, (0.55, 0.20, 0.25), True),
    MerchantCategory("gaming", 0.06, 300.0, (0.85, 0.12, 0.03), True),
    MerchantCategory("education", 0.05, 6500.0, (0.40, 0.30, 0.30), True),
    MerchantCategory("healthcare", 0.07, 1200.0, (0.60, 0.30, 0.10), False),
    MerchantCategory("fashion", 0.10, 1900.0, (0.50, 0.40, 0.10), True),
    MerchantCategory("electronics", 0.09, 8500.0, (0.30, 0.55, 0.15), True),
)
