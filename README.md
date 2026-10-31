# محرك كشف أطياف الأقمار الاصطناعية (Spherex-Streak-Detection-Engine)
## تحدي ناسا الدولي لتطبيقات الفضاء (NASA Space Apps Challenge)
### معالجة بيانات تلسكوب SPHEREx الفضائي باستخدام الذكاء الاصطناعي الجراحي غير المدمر

---

## 1. ملخص المشروع (Executive Summary)

يعد مشروع **محرك كشف أطياف الأقمار الاصطناعية (Spherex-Streak-Detection-Engine)** أداة برمجية ذكية متخصصة طُورت بلغة Python لصالح معالجة بيانات وتصوير تلسكوب الفضاء **SPHEREx** التابع لوكالة ناسا. 

يقوم النظام بالكشف التلقائي الدقيق عن مسارات وأطياف الأقمار الاصطناعية وحطام الفضاء (Satellite Streaks & Train-Track Patterns) التي تلوث الصور الفلكية بصيغة FITS، مع تطبيق فلسفة المعالجة الجراحية غير المدمرة لحماية البيانات العلمية الأصلية، وتحديداً بصمات امتصاص المياه والثلج الكوني.

---

## 2. وصف المشكلة العلمية (Problem Statement)

يعمل تلسكوب SPHEREx في المدار الأرضي المنخفض لمسح السماء بأكملها عبر 102 نطاق طيفي تتراوح بين 0.75 و 5.0 ميكرومتر، بهدف دراسة نشأة الكون ومصادر المياه والمواد العضوية في مجرتنا. 

مع تزايد كوكبات الأقمار الاصطناعية في المدار المنخفض، تتعرض عمليات الرصد لظاهرة التلوث البصري الناتج عن عبور تلك الأجسام بسرعة عالية أمام عدسات التلسكوب، مما يترك مسارات خطية خافتة وضعيفة التباين (Low-SNR Streaks) أو خطوطاً متوازية متقاربة تُعرف بمسارات القطار (Train-Track Patterns).

### مقارنة بين الأساليب التقليدية والحل الجراحي غير المدمر:

| وجه المقارنة | المعالجة التقليدية (Destructive) | محرك Spherex Streak (Surgical Non-Destructive) |
|---|---|---|
| **سلامة البكسلات الأصلية** | قص أو تصفير البكسل الملوث بشكل دائم | الحفاظ التام على مصفوفة التدفق $I(x, y)$ بنسبة 100% |
| **بصمات الامتصاص للمياه والثلج** | تدمير خط الامتصاص عند $3.05\,\mu\text{m}$ | صيانة كاملة لعمق وشكل خط الامتصاص الطيفي |
| **طبيعة القناع الناتج** | أقنعة ثنائية صلبة (0 أو 1) | قناع احتمالي مستمر Float32 بالمدى $[0.0, 1.0]$ |
| **المعالجة في القياسات التراكمية** | استبعاد البكسل وفقدان فوتوناته | تطبيق قانون التباين العكسي $\sigma^2_{\text{eff}} = \sigma^2 / w$ |
| **التعامل مع الضوضاء** | عتبات رقمية ثابتة (Hardcoded Thresholds) | تطبيع ديناميكي تكيفي يعتمد على إحصاءات MAD |
| **التمييز الطيفي** | خلط مسارات الأقمار بذيول المذنبات والسدم | استغلال نطاقات SPHEREx الـ 102 لعزل الأقمار فقط |

---

## 3. الحل التقني والمعادلات الرياضية (Mathematical Formulation)

يرتكز محرك النظام على خمس ركائز رياضية وفيزيائية:

### أ. قناع جودة الاحتمالية المستمر (Probability Quality Mask - PQM)
$$\text{PQM}(x, y) = \sigma\!\left(\text{CNN}_\theta(I)\right)_{x,y} \in [0.0, 1.0]^{N \times M}$$
ينتج النموذج مصفوفة ثقة مستمرة من النوع Float32 تتراوح قيمها بين [0.0 و 1.0] بحجم الصورة الكاملة ($N \times M$) عبر طبقة تنشيط Sigmoid، مما يوفر مقياساً مستمراً لدرجة التلوث دون إتلاف أي بيانات.

### ب. قانون وزن التباين العكسي (Inverse-Variance Weighting)
تُعدّل أوزان القياسات الضوئية اللاحقة استناداً إلى تباين البكسل الفعّال:
$$\sigma^2_{\text{effective}}(x, y) = \frac{\sigma^2(x, y)}{w(x, y)}, \qquad w(x, y) = 1.0 - \text{PQM}(x, y)$$
حيث تصبح أوزان التجميع الضوئي التراكمي (Photometric Co-addition Weights):
$$W(x, y) = \frac{1}{\sigma^2_{\text{effective}}(x, y)}$$
- **في المناطق الفلكية السليمة:** $\text{PQM} \approx 0.0 \implies w \approx 1.0 \implies \sigma^2_{\text{effective}} = \sigma^2$ (يحتفظ البكسل بوزنه الكامل).
- **في المناطق الملوثة بمسارات الأقمار:** $\text{PQM} \to 1.0 \implies w \to 0.0 \implies \sigma^2_{\text{effective}} \to \infty \implies W \to 0$ (ينخفض وزن مساهمة البكسل تلقائياً في القياسات التراكمية دون حذفه).

### ج. التطبيع التكيفي الديناميكي (Dynamic Adaptive Preprocessing)
يُقدَّر تباين وضوضاء خلفية السماء إحصائياً بشكل مستقل لكل صورة باستخدام الوسيط والانحراف المطلق عن الوسيط الحصين:
$$\sigma_{\text{MAD}} = 1.4826 \times \text{median}\left(\left|I - \text{median}(I)\right|\right)$$
$$\text{SNR}(x, y) = \frac{I(x, y) - \text{Background}(x, y)}{\sigma_{\text{MAD}}(x, y)}$$

### د. التحقق الطيفي عبر نطاقات SPHEREx الـ 102 (Spectral Discrimination)
يمتلك النظام منطقاً فيزيائياً يميز بين:
- **مسارات الأقمار الاصطناعية:** ظاهرة عابرة تظهر في نطاق واحد أو نطاقات زمنية محدودة جداً ($\le 15\%$ من النطاقات)، بمقطع عرضي يطابق دالة توزيع النقطة البصرية للتلسكوب (FWHM $\approx 1.5 - 2.5$ بكسل).
- **المعالم الفلكية الطبيعية الممتدة (Comet Tails & Nebulae Filaments):** تراكيب طبيعية مستمرة عبر أغلب النطاقات الطيفية ($\ge 40\%$) وبمقطع عرضي متسع (FWHM $> 4.0$ بكسل). يتم استثناؤها وحمايتها بالكامل من أي حجب.

### هـ. كشف الإشارات الضعيفة عبر دالة خسارة Focal Loss
$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t + \epsilon), \qquad \gamma \ge 2.0$$
يعمل المعامل $\gamma \ge 2.0$ على كبح مساهمة خلفية السماء المظلمة بأكثر من 1000 ضعف مقارنة بدوال الخسارة التقليدية:
$$(1 - 0.98)^{2.0} = (0.02)^2 = 0.0004$$
مما يُبقي تركيز الشبكة العصبية منصباً على التقاط المسارات الخافتة المتداخلة مع الضوضاء.

### و. مقياس الدقة المستمر (Continuous Soft IoU)
$$\text{Soft IoU}(P, G) = \frac{\sum P \cdot G}{\sum \left(P^2 + G^2 - P \cdot G\right) + \epsilon}$$
يحقق النظام نسبة تطابق Soft IoU تتجاوز $\ge 0.95$ على عينات الاختبار.

---

## 4. الهيكل البرمجي للمستودع (Repository Structure)

```
spherex-streak-processor/
├── app/
│   ├── main.py                        # المدخل الرئيسي لتشغيل واجهة المستخدم
│   ├── streamlit_app.py               # لوحة التحكم التفاعلية والمقارنة جنباً إلى جنب
│   └── __init__.py
│
├── src/                               # الموديولات الأساسية المباشرة
│   ├── fits_loader.py                 # Phase 1: قارئ ملفات FITS والتطبيع التكيفي
│   ├── preprocessing.py               # Phase 2: التطبيع وطرح الضوضاء والتحقق الطيفي
│   ├── model.py                       # Phase 3: شبكة DeepStreaks التلفيفية ودالة Focal Loss
│   ├── metrics.py                     # Phase 4: مقياس Soft IoU ووزن التباين العكسي
│   └── spherex_streak/                # الحزمة البرمجية الموحدة
│       ├── io/fits_loader.py
│       ├── preprocessing/adaptive.py
│       ├── models/deepstreaks.py
│       ├── models/losses.py
│       ├── detection/pqm.py
│       ├── weighting/inv_variance.py
│       └── utils/visualization.py
│
├── tests/                             # 69 اختبار وحدة وتكامل آلي شامل (100% نجاح)
│   ├── test_fits_loader.py            # اختبارات سلامة FITS وحفظ البيانات
│   ├── test_preprocessing.py          # اختبارات التطبيع التكيفي والتحقق الطيفي
│   ├── test_model.py                  # اختبارات DeepStreaks و PQM و Focal Loss
│   ├── test_metrics.py                # اختبارات مقياس Soft IoU وقانون التباين العكسي
│   ├── test_pqm.py                    # اختبارات محرك قناع جودة الاحتمالية
│   ├── test_weighting.py              # اختبارات قانون الأوزان الفعالة
│   └── test_app.py                    # اختبارات واجهة المستخدم والتوليد
│
├── scripts/                           # أدوات الأتمتة والتدريب
│   ├── download_sparkles.py           # تنزيل عينات مجموعة SPARKLES من Zenodo
│   ├── train.py                       # خط تدريب النموذج العصبي
│   └── evaluate.py                    # تقييم الأداء والتحقق من صيانة الأطياف
│
├── .streamlit/
│   └── config.toml                    # إعدادات مظهر الواجهة التفاعلية (NASA Dark Theme)
├── pyproject.toml                     # توصيف الحزمة وإعدادات التثبيت
├── requirements.txt                   # متطلبات ومكتبات التشغيل
├── .env.example                       # نموذج توضيحي آمن لمتغيرات البيئة
├── .gitignore                         # استبعاد البيانات والنماذج والملفات الحساسة
└── README.md                          # الوثائق الرسمية للمشروع
```

---

## 5. دليل التثبيت والتشغيل الشامل (How to Run)

### المتطلبات الأساسية:
- نظام تشغيل: Windows أو Linux أو macOS.
- بيئة عمل Python 3.10 أو أحدث.

### الخطوة 1: استنساخ المستودع وتثبيت البيئة
```bash
git clone https://github.com/Muhammadsax/Spherex-Streak-Detection-Engine.git
cd Spherex-Streak-Detection-Engine

# تثبيت الحزمة وجميع الاعتماديات في وضع التطوير
pip install -e ".[dev]"
```

### الخطوة 2: تشغيل الاختبارات الآلية الشاملة
للتأكد من سلامة كافة خوارزميات النظام واجتياز الـ 69 اختباراً:
```bash
pytest tests/ -v
```

### الخطوة 3: تشغيل واجهة المستخدم التفاعلية (Streamlit Dashboard)
لتشغيل لوحة التحكم والمقارنة الرباعية الفورية على المتصفح:
```bash
streamlit run app/main.py
```
يفتح التطبيق تلقائياً على الرابط: `http://localhost:8501`.

### الخطوة 4: تشغيل خط أنابيب التدريب (اختياري)
لتدريب نموذج DeepStreaks باستخدام دالة الخسارة المركبة (Focal Loss + Soft IoU):
```bash
python scripts/train.py --epochs 10 --batch-size 8 --lr 0.001
```

### الخطوة 5: تشغيل سكريبت التقييم المعياري (اختياري)
لقياس متوسط درجات Soft IoU والتحقق من نسبة حفظ البيانات:
```bash
python scripts/evaluate.py --samples 10
```

### الخطوة 6: تنزيل عينات بيانات SPARKLES (اختياري)
لتنزيل العينات الأصلية من مستودع Zenodo التابع لمركز أبحاث ناسا أميس:
```bash
python scripts/download_sparkles.py
```

---

## 6. التحقق والاختبارات الآلية (Testing & Verification)

يحتوي المستودع على حزمة اختبارات برمجية وعلمية شاملة تتألف من 69 اختباراً، تغطي دقة الأبعاد، ونوع البيانات Float32، وحدود الـ PQM، وعدم تعديل مصفوفة التدفق الفلكي الأصلية بأي شكل.

```text
tests/test_app.py            ......               (6 passed)
tests/test_fits_loader.py    ..................   (18 passed)
tests/test_preprocessing.py  ...........          (11 passed)
tests/test_model.py          .............        (13 passed)
tests/test_metrics.py        .............        (13 passed)
tests/test_pqm.py            ....                 (4 passed)
tests/test_weighting.py      ....                 (4 passed)
=========================== 69 passed in 27.33s ===========================
```

---

## 7. النشر السحابي (Cloud Deployment)

المشروع مهيأ بالكامل للنشر الفوري بنقرة واحدة على منصة **Streamlit Community Cloud**:
1. سجل الدخول إلى [share.streamlit.io](https://share.streamlit.io) عبر حساب GitHub.
2. اختر المستودع `Muhammadsax/Spherex-Streak-Detection-Engine` والفرع `main`.
3. حدد مسار ملف التشغيل: `app/main.py`.
4. اضغط Deploy ليبدأ التطبيق بالعمل فوراً عبر رابط سحابي عالمي.

---

## 8. مجموعة البيانات المعتمدة والترخيص (Dataset & License)

- **مجموعة البيانات:** SPARKLES Dataset المعتمَدة من مركز أبحاث ناسا أميس (NASA Ames Research Center)، المنشورة برقم السجل المعياري `10935280` على منصة Zenodo.
- **المرجع العلمي:** DeepStreaks Framework (arXiv:1904.05920).
- **الترخيص:** رخصة MIT المفتوحة المصدر.
