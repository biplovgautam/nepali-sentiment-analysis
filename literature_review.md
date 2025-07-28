
### **1. Origin and Domain Adaptation**

* **Sentiment analysis** originally developed in general areas like **movie or product reviews**.
* **Pang and Lee (2008)** highlighted a key issue: **models trained in one domain don’t transfer well** to another. For example, something seen as positive in a book review might be negative in a movie review.
* This illustrates the **importance of domain adaptation**—adapting tools and models to specific contexts.

---

### **2. Domain Mismatch in Financial Text**

* **Loughran and McDonald (2011)** found that **general sentiment dictionaries misclassify financial terms**.
* Words like **“liability”** or **“tax”** aren’t inherently negative in finance, but general lexicons labeled them that way.
* As a solution, they created a **financial-specific sentiment dictionary** to improve accuracy.

---

### **3. Value of Domain-Specific Resources**

* **Wang et al. (2013)** showed that a **finance-specific lexicon** could predict **firm risk** with performance rivaling more complex machine learning models.
* This reinforced the idea that **domain relevance is crucial** for accurate sentiment analysis.

---

### **4. Rise of Classical Machine Learning**

* Researchers like **Schumaker and Chen (2009)** used **Naive Bayes, SVMs, and decision trees** to analyze financial texts—showing traditional ML worked reasonably well.
* **Palmer et al. (2020)** confirmed that **general dictionaries are simple but less accurate** than models trained on finance-specific data.
* For example, **Huang et al. (2014)** achieved over **80% accuracy** using a Naive Bayes model trained on annotated financial reports, compared to \~62% using a standard dictionary.

---

### **5. Advances in Deep Learning**

* Deep learning brought major improvements:

  * **FinBERT (Araci, 2019)** is a version of BERT trained on financial texts, yielding **state-of-the-art accuracy**.
  * **Zhu et al. (2022)** found that **transformer models** like BERT could **double or triple the accuracy** of older methods.
* These models are powerful because they can **learn from massive data with minimal human labeling**, but they require:

  * **Large datasets**
  * **High computational power**
  * **Significant fine-tuning**, making them harder to use in real-time or on limited systems.



## Refrences

- Pang and Lee (2008)
– Domain adaptation in sentiment analysis.

- Schumaker and Chen (2009)
– Applied traditional ML (Naive Bayes, SVM, decision trees) to financial text.

- Loughran and McDonald (2011)
– Developed a financial sentiment lexicon highlighting domain mismatch.

- Wang et al. (2013)
– Used a finance-specific lexicon to predict firm risk.

- Huang et al. (2014)
– Achieved 80.9% accuracy using Naive Bayes on analyst reports.

- Araci (2019)
– Introduced FinBERT, a BERT model pretrained for financial sentiment tasks.

Palmer et al. (2020)
– Reviewed sentiment analysis approaches; favored domain-specific ML.

Zhu et al. (2022)
– Showed that transformer models outperform traditional methods.