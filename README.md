# 🌌 محرك كشف أطياف الأقمار الاصطناعية — Spherex-Streak-Detection-Engine
### NASA Space Apps Challenge | SPHEREx Satellite Streak Detection & Surgical Processing

[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)]()
[![Astropy](https://img.shields.io/badge/Astropy-5.3+-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

نظام ذكاء اصطناعي ومعالجة صور فلكية خفيف الوزن، مصمم لمعالجة صور تلسكوب **NASA SPHEREx** الفضائي لكشف أطياف ومسارات الأقمار الاصطناعية (Satellite Streaks & Train-Track Patterns) بطريقة **جراحية غير مدمرة** تحافظ بنسبة 100% على البيانات الطيفية الأصلية وبصمات امتصاص المياه والثلج الكونية دون حذف أو تصفير أي بكسل.

---

## 🎯 الأهداف والمعايير الأساسية (Core Technical Guardrails)

1. **معالجة جراحية غير مدمرة (Surgical Non-Destructive):** الحفاظ التام على التدفق الفلكي $I(x, y)$ وبصمات الامتصاص للمياه والثلج عند $3.05\,\mu\text{m}$.
2. **قناع جودة الاحتمالية (PQM Matrix):** مصفوفة ثقة مستمرة من النوع `Float32` بالمدى $[0.0, 1.0]$ ونفس حجم الصورة الأصلي $(N \times M)$ عبر دالة `Sigmoid`.
3. **قانون وزن التباين العكسي (Inverse-Variance Weighting):**
   $$\sigma^2_{\text{effective}}(x, y) = \frac{\sigma^2(x, y)}{w(x, y)}, \qquad w(x, y) = 1.0 - \text{PQM}(x, y)$$
4. **تطبيع تكيفي ديناميكي (Dynamic Preprocessing):** تقدير إحصائي حصين لضوضاء الخلفية عبر $\sigma_{\text{MAD}} = 1.4826 \times \text{MAD}$ بدون أي عتبات صلبة ثابتة (`no hardcoded thresholds`).
5. **التحقق الطيفي عبر نطاقات SPHEREx الـ 102:** التمييز الفيزيائي بين مسارات الأقمار العابرة والمعالم الفلكية الطبيعية الممتدة (Comet tails, Nebulae filaments).
6. **دالة خسارة Focal Loss ($\gamma \ge 2.0$):** لكشف الإشارات الضعيفة القريبة من أرضية الضوضاء (Low-SNR faint streaks near noise-floor).
7. **مقياس التقييم المستمر Soft IoU:** استهداف تحقيق نسبة دقة $\ge 0.95$.
8. **واجهة تفاعلية فورية (Interactive Streamlit UI):** مقارنة جنباً إلى جنب للصور الأصلية والـ PQM وخريطة التباين الفعال.

---

## 🏗️ مراحل التنفيذ الخمس (5-Phase Architecture)

```
spherex-streak-processor/
├── app/
│   ├── streamlit_app.py               ← واجهة العرض التفاعلية الشاملة
│   └── main.py                        ← المدخل الرئيسي لتشغيل الواجهة
├── src/                               ← الموديولات الأساسية المباشرة
│   ├── fits_loader.py                 ← قارئ FITS الآمن وتطبيع النسب المئوية
│   ├── preprocessing.py               ← التطبيع التكيفي والتحقق عبر 102 نطاق
│   ├── model.py                       ← نموذج DeepStreaks CNN ودالة Focal Loss
│   ├── metrics.py                     ← مقياس Soft IoU ووزن التباين العكسي
│   └── spherex_streak/                ← حزمة بايثون الموحدة
│       ├── io/fits_loader.py
│       ├── preprocessing/adaptive.py
│       ├── models/deepstreaks.py
│       ├── models/losses.py
│       ├── detection/pqm.py
│       ├── weighting/inv_variance.py
│       └── utils/visualization.py
├── tests/                             ← 69 اختبار آلي شامل (100% نجاح)
│   ├── test_fits_loader.py            (18 tests)
│   ├── test_preprocessing.py          (11 tests)
│   ├── test_model.py                  (13 tests)
│   ├── test_metrics.py                (13 tests)
│   ├── test_pqm.py                    (4 tests)
│   ├── test_weighting.py              (4 tests)
│   └── test_app.py                    (6 tests)
└── scripts/
    ├── download_sparkles.py           ← تنزيل عينات SPARKLES من Zenodo
    ├── train.py                       ← تدريب النموذج بدالة Focal + Soft IoU
    └── evaluate.py                    ← تقييم الأداء والتحقق من حفظ البيانات
```

---

## ⚡ التشغيل السريع (Quickstart)

### 1. تثبيت الحزمة
```bash
git clone https://github.com/Muhammadsax/Spherex-Streak-Detection-Engine.git
cd Spherex-Streak-Detection-Engine
pip install -e ".[dev]"
```

### 2. تشغيل الاختبارات الآلية
```bash
pytest tests/ -v
# النتيجة: 69 passed in ~9s ✅
```

### 3. إطلاق واجهة المستخدم التفاعلية (Streamlit Dashboard)
```bash
streamlit run app/main.py
```
سيعمل التطبيق محلياً على المتصفح عبر الرابط: `http://localhost:8501`.

---

## 🔬 الخوارزميات والمعادلات الرياضية (Mathematical Laws)

### 1. Continuous Soft IoU (Continuous Jaccard Index)
$$\text{Soft IoU}(P, G) = \frac{\sum_{i} P_i \cdot G_i}{\sum_{i} \left(P_i^2 + G_i^2 - P_i \cdot G_i\right) + \epsilon}$$
- يعامل الـ PQM كقيم احتمالية مستمرة في $[0.0, 1.0]$.
- يعطي $1.0$ في حالة التطابق التام، ويحقق الهدف التقني $\ge 0.95$.

### 2. Inverse-Variance Weighting Law
$$\sigma^2_{\text{effective}}(x, y) = \frac{\sigma^2(x, y)}{w(x, y)}, \qquad W(x, y) = \frac{1}{\sigma^2_{\text{effective}}(x, y)}$$
- البكسلات الملوثة بمسارات الأقمار الصناعية ($w \to 0$) تصبح $\sigma^2_{\text{effective}} \to \infty$ وبالتالي ينخفض وزنها الضوئي $W \to 0$ تلقائياً.
- البكسلات السليمة تحتفظ بـ $\sigma^2$ ووزنها كاملاً.

### 3. Focal Loss for Low-SNR Signals
$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t + \epsilon), \qquad \gamma \ge 2.0$$
- يكبح مساهمة خلفية السماء المظلمة السهلة بأكثر من 1000 ضعف مقارنة بالـ BCE، مما يركز التدرج على المسارات الخافتة.

---

## 🚀 النشر السحابي (Cloud Deployment)

التطبيق مهيأ بالكامل للنشر الفوري بنقرة واحدة على:
- **Streamlit Community Cloud:** باختيار مستودع GitHub وتحديد الملف `app/main.py`.
- **Hugging Face Spaces:** باختيار SDK: Streamlit.
- ملف التهيئة `.streamlit/config.toml` مدمج ومضبوط بالسمة الفضائية الداكنة (NASA Dark Theme).

---

## 👥 المساهمون وفريق العمل
- **Project:** Spherex-Streak-Detection-Engine
- **Event:** NASA Space Apps Challenge
- **Focus:** NASA SPHEREx Mission & SPARKLES Dataset
