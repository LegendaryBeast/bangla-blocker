#!/usr/bin/env python3
"""
generate_allow_500.py
Generates 500 authentic, clinically realistic routine healthcare ALLOW scenarios
spanning 10 major operational hospital departments in Bangladesh:
1. OPD & Specialist Appointments
2. Diagnostic Imaging & Radiology
3. Pathology & Lab Tests
4. Emergency, Critical Care & Ambulance
5. Inpatient Admission, Wards & Discharge
6. Hospital Pharmacy & Medication Guidance
7. Billing, Payments & Health Insurance
8. Visiting Hours & Hospital Facilities
9. Vaccination & Preventive Healthcare
10. Telemedicine, Blood Bank & Support Services
"""

import csv
import os

scenarios = []

def add(en, bn, bg, cm, cat_name):
    idx = len(scenarios) + 1301
    scenarios.append({
        "id": f"SBH-HW-{idx}",
        "domain": "healthcare",
        "security_intent": "B0_legitimate_security",
        "hard_negative_type": "routine_healthcare",
        "decision": "ALLOW",
        "risk_level": "none",
        "code_mix": cm,
        "banglish": bg,
        "bangla": bn,
        "english": en,
        "note": f"routine healthcare inquiry -- {cat_name}"
    })

print("Building 500 routine healthcare scenarios...")
