# Hospital Readmission Prediction | Casimir Uhlig | Tomorrow University - ML-Classification Project

## 🏥 Context
The German healthcare system faces mounting financial pressure, with state insurance companies paying **€102 billion in 2024** for hospital care. [https://www.vdek.com/presse/daten/d_versorgung_leistungsausgaben.html] A significant portion of these costs stems from preventable readmissions and suboptimal discharge decisions.

Currently, German hospitals rely primarily on clinical judgment when deciding patient discharge and post-care monitoring. This project demonstrates how machine learning can enhance these decisions by identifying which patients can safely return home versus those requiring extended monitoring—while maintaining strict **GDPR compliance**.

* **Impact:** Reduced costs, improved patient outcomes, and lives saved through evidence-based discharge decisions.

---

## 🎯 Problem Statement
Given a diabetes patient's clinical history, demographics, and treatment data at hospital discharge:
* **Predict:** Readmission status within 24 hours of discharge.
* **Goal:** Enable targeted post-discharge monitoring that reduces unnecessary readmissions while ensuring high-risk patients receive appropriate care.

### Project Specifications
* **Type:** Multiclass Classification
* **Target Variable:**
    1. No readmission
    2. Readmission within 30 days
    3. Readmission after 30 days
* **Prediction Horizon:** Data available at discharge (T=0)
* **Prediction Window:** 30-day readmission risk
* **Execution:** Within 24 hours pre-discharge for care coordination

---

## 💡 Hypothesis & Intended Impact

### Stakeholder Impact Table
| Impact Type | Direct Impact | Indirect Impact |
| :--- | :--- | :--- |
| **Benefit** | **Hospital Staff:** Optimized bed management and clearer discharge workflows. | **Taxpayers/Insurers:** Long-term stabilization of healthcare premiums due to lower system costs. |
| **Harm / Risk** | **Vulnerable Patients:** Risk of "algorithmic bias" where certain demographics are flagged for early discharge incorrectly. | **Medical Staff Autonomy:** Potential "automation bias" where doctors stop questioning the AI, leading to skill atrophy. |

> **Key Benefits:** Optimized resource allocation, reduced healthcare costs, and better patient outcomes through targeted intervention. <br>
> **Potential Harms:** Discriminatory predictions affecting vulnerable populations or inappropriate early discharge leading to adverse health events.

---

## 📊 Dataset Discovery & Selection
Why this Dataset, this DataSet is openly available to Download, other Hospital DataSets you have to Request Access or be Part of Specific Institutions.

### Chosen Dataset
**Diabetes 130 US hospitals for years 1999-2008**
* Dataset download: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
* Data set by: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.
