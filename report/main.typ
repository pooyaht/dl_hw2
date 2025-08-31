#import     "@preview/problemst:0.1.2": pset

#show: pset.with(
  class: "Deep Learning",
  student: "Pooya Hatami - 4031334011",
  title: "Project_2",
  date: datetime(
    year: 2025,
    month: 8,
    day: 11,
    ),
)

#let my_heading(title, level: 1) = [
  #heading(numbering: none, level: level, title)
]


#set text(12pt)
#set text(lang: "fa", font: "vazirmatn")
#show link: it => {
  text(blue, underline(it))
}
	
#set enum(indent: 1em) 
#set enum(numbering: "۱.") 
#set list(indent: 1em) 

#set table( stroke: none, gutter: 0.2em, fill: (x, y) => if x == 0 or y == 0 { luma(50%) }, inset: (right: 1.5em), ) 
#show table.cell: it => { if it.x == 0 or it.y == 0 { set text(white) 
strong(it) } else { it } } 

#set figure(numbering: "۱")
#show figure.where(kind: table): set figure(supplement: "جدول")
#show figure.where(kind: image): set figure(supplement: "شکل")


#set footnote(numbering: "۱")

#set par(justify: true)

#my_heading("استفاده از مدل YOLO")
در این روش از مدل YOLOv11L برای inference بدون هیچ گونه تغییر استفاده شده و همچنین مقدار حد نصاب اطمینان#footnote[Confidence Threshold] برای در نظر گرفتن جعبه فراگیر#footnote[Bounding Box] برابر با ۰.۷ در نظر گرفته شده است. با توجه به توزیع متفاوت دو داده تست، مدل سنگین و حدنصاب بالایی در نظر گرفته شده است تا به نتیجه مطلوب دست یابییم.
#figure(
  image("./cat_and_dog_yolo11l.jpg", width: 100%),
  caption: "نتیجه مدل YOLOv11L بر روی عکس cat_and_dog"
)

#figure(
  image("./random_yolo11l.jpg", width: 100%),
  caption: "نتیجه مدل YOLOv11L بر روی عکس random"
)

#align(center)[
    #box(
        width: 80%,
        height: auto,
        fill: rgb(240, 240, 255),
        radius: 5pt,
        stroke: 2pt + rgb(100, 100, 255),
        inset: 20pt,
    )[
        #par(justify: false)[
            #text(size: 16pt, weight: "bold")[لینک نوت‌بوک YOLO در Colab ]
        ]
        #v(10pt)
        #link("https://colab.research.google.com/drive/1D4iba_66iWNj_Qo5m_aHgJWLTz-10a3O?usp=sharing")[
            #image("./colab-badge.svg")
        ]
    ]
]

#my_heading("استفاده از مدل سفارشی")
در این قسمت از یک مدل سفارشی با الهام از معماری YOLOv2 و با پشتوانه ResNet50 استفاده می‌شود به اینصورت که وجود سگ و گربه و همچنین موقعیت سگ و گربه در یک مرحله پیش‌بینی می‌شود.

#my_heading("مجموعه‌داده", level: 2) 
برای آموزش از کلاس سگ و گربه مجموعه‌داده Coco2017 استفاده شد. قسمت‌های آموزش و اعتبارسنجی با هم ادغام شده و سپس به صورت هوشمندانه‌تر جدا می‌شوند. کلیه کدهای مربوط به این قسمت در فایل `coco_downloader.py` موجود می‌باشد.

#block[
#set par(justify: false)
#figure(
  table(
    columns: 5,
    [مجموعه‌داده], [تعداد تصاویر شامل سگ و گربه],[تعداد تصاویر شامل فقط سگ],[تعداد تصاویر شامل فقط گربه], [مجموع],
    [Coco2017 (train+valid)], [۲۲۰], [۴۳۴۲], [۴۰۷۸], [*۸۶۴۰*]
  ),
  caption: [مقایسه مجموعه‌داده‌های مورد استفاده]
)
]
برای آموزش و اعتبارسنجی تمامی تصاویر شامل سگ و گربه(به علت تعداد کم)‌ انتخاب می‌شوند سپس هر داده‌های هر کلاس بر اساس تعداد Bounding Box مرتب شده و ۲۵۰۰ داده اول از هرکلاس انتخاب می‌شوند. با اینکار از تمام عکس‌های شامل چندین سگ یا گربه(که تعداد کمی هم دارند) استفاده می‌شود.

#my_heading("آماده‌سازی داده جهت آموزش", level: 2)
در این قسمت داده‌ها به دو قسمت آموزش و ارزیابی که نسبت آن توسط پارامتر `val_ratio` تعیین می‌شود، تقسیم می‌شوند. 

#figure(
  image("./val_train_distribution.png", width: 100%),
  caption: "توزیع تعداد لیبل ها در داد‌ه‌های آموزش و ارزیابی(نسبت ۰.۲)"
)
پیش‌پردازش ثابت انجام‌شده روی همه داده‌ها عبارتند از:
- تغییر ابعاد عکس‌ها به (۲۲۴، ۲۲۴) متناظر ResNet50 از پیش آموزش دیده#footnote[Pretrained] برروی ImageNet 
- نرمال‌سازی عکس‌ها با میانگین و انحراف از معیار مدل ResNet50 

همچنین برای جلوگیری از بیش‌برازش#footnote[Overfitting] از Augmentation های مختلف نیز استفاده شده است که استفاده و احتمال اعمال آن‌ها توسط پارامتر قابل تنظیم می‌باشد.
```python
def get_train_augmentations(target_size=(224, 224), p=0.5):
    transforms_list = [
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15,  
                val_shift_limit=10, p=1.0),
        ], p=p),

        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=p * 0.2),

        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=p * 0.7
        ),

        A.RandomSizedBBoxSafeCrop(
            height=target_size[0],
            width=target_size[1],
            erosion_rate=0.2,
            p=p * 0.3
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(transforms_list,
        bbox_params=A.BboxParams(format='albumentations',           
        label_fields=['class_labels']))

```
به دلیل اختلاف بین توزیع عکس‌های آموزش با دو عکس تست(عکس‌های آموزش عکس از حیوانات در محیط‌های طبیعی و با کیفیت کم و دو عکس تست در استودیو و با کیفیت بالا می‌باشند)، با یک روش ساده، به صورت تصادفی و با احتمال قابل تنظیم، سعی می‌شود پس‌زمینه عکس‌ها به سفید تغییر یابد که در ادامه خروجی آن آورده‌ شده است:

#figure(
  image("./remove_bg.png", width: 100%),
  caption: "حذف پس‌زمینه عکس‌ها"
)

#pagebreak()

#my_heading("معماری مدل", level: 2)
معماری مدل سفارشی `SimpleResNetYOLO` شامل دو قسمت اصلی می‌باشد:

#my_heading("شبکه پشتیبان (Backbone)", level: 3)
از مدل ResNet50 پیش‌آموزش‌دیده بر روی مجموعه‌داده ImageNet به عنوان شبکه پشتیبان استفاده شده است. برای سازگاری با معماری YOLO، لایه‌های fully connected انتهایی(دو لایه آخر) حذف شده و فقط قسمت convolutional باقی مانده است. 

برای پشتیبانی از grid size های مختلف، تنظیمات خاصی روی لایه چهارم ResNet50 انجام شده است:
- برای grid size ۱۴×۱۴ با target size ۲۲۴: stride تبدیل به ۱ و dilation به ۲ تغییر می‌یابد
- برای grid size ۲۸×۲۸ با target size ۴۴۸: تنظیمات مشابه اعمال می‌شود

این تنظیمات باعث می‌شود که feature map خروجی ResNet50 با اندازه مطلوب تولید شود.

#my_heading("سر پیش‌بینی (Prediction Head)", level: 3)
سر پیش‌بینی شامل چهار لایه convolutional است که به صورت متوالی عمل می‌کنند:

+ *لایه اول*: کاهش کانال‌ها از ۲۰۴۸ به ۲۵۶ با kernel سایز ۱ 
+ *لایه دوم*: convolution با kernel سایز ۳، padding برابر ۲ و dilation برابر ۲
+ *لایه سوم*: convolution با kernel سایز ۱ برای ترکیب feature ها
+ *لایه آخر*: تولید خروجی نهایی بدون activation

سه لایه اول دارای Batch Normalization، ReLU و Dropout با نرخ ۰.۴ هستند. لایه آخر bias خود را به گونه‌ای مقداردهی اولیه می‌کند که احتمال اولیه objectness برابر ۰.۰۱ باشد.

خروجی نهایی برای هر anchor box شامل:
- ۴ مختصات bounding box (x, y, w, h)
- ۱ امتیاز اطمینان (objectness score)  
- ۲ امتیاز کلاس (سگ و گربه)

با grid size ۱۴×۱۴ و ۵ anchor box، تعداد کل پیش‌بینی‌ها برابر با $۱۴×۱۴×۵×۷ = ۶۸۶۰$ خواهد بود.

#my_heading("استراتژی آموزش", level: 3)
برای بهینه‌سازی فرآیند آموزش، از استراتژی "Progressive Unfreezing" استفاده شده است:

+ *مرحله اول* (تا epoch ۱۲): تمام لایه‌های backbone منجمد، نرخ یادگیری ۱e-۳
+ *مرحله دوم* (epoch ۱۲ تا ۲۴): دو لایه آخر backbone آزاد، نرخ یادگیری ۱e-۵  
+ *مرحله سوم* (epoch ۲۴ به بعد): تمام لایه‌های backbone آزاد، نرخ یادگیری ۱e-۶

این استراتژی باعث تطبیق تدریجی شبکه پشتیبان با داده‌های هدف می‌شود.

#my_heading("Anchor Box Generation", level: 3)
برای تولید anchor box ها از الگوریتم K-Means با ۳ خوشه بر روی ابعاد bounding box های داده‌های آموزش استفاده شده است. سپس دو anchor اضافی با نسبت‌های ۱۶:۹ و ۹:۱۶ بر اساس بزرگترین anchor محاسبه شده اضافه می‌شوند.

Anchor box های نهایی محاسبه شده:
```
[[ 2.27  2.54]
 [ 8.25  4.64] 
 [ 4.64  8.25]
 [ 5.78  6.62]
 [10.18 10.93]]
```

#my_heading("تنظیمات آموزش", level: 2)
پارامترهای نهایی آموزش بر اساس آزمایش‌های انجام شده:

#block[
#set par(justify: false)
#figure(
  table(
    columns: 2,
    [پارامتر], [مقدار],
    [Grid Size], [۱۴×۱۴],
    [Target Size], [۴۴۸×۴۴۸],
    [Batch Size], [۸],
    [Epochs], [۳۰],
    [Dropout Rate], [۰.۴],
    [Weight Decay], [۱e-۳],
    [Coord Weight], [۱.۰],
    [Augmentation Strength], [۰.۵],
    [Background Removal Probability], [۰.۵]
  ),
  caption: [پارامترهای نهایی آموزش]
)
]

#my_heading("Loss Function", level: 3)
تابع هزینه ترکیبی شامل چهار بخش اصلی است:

+ *XY Loss*: MSE loss برای مختصات مرکز bounding box
+ *WH Loss*: MSE loss برای ابعاد bounding box (در فضای log)
+ *Objectness Loss*: Focal Loss برای تشخیص وجود object با α=۰.۲۵ و γ=۲.۰
+ *Classification Loss*: Binary Cross Entropy برای تشخیص نوع حیوان

```python
total_loss = coord_weight * (xy_loss + wh_loss) + objectness_loss + cls_loss
```

#my_heading("نتایج آموزش", level: 2)
مدل به مدت ۳۰ epoch آموزش داده شد. در طول آموزش، بهترین validation loss در epoch مناسب ذخیره شد:

#figure(
  image("./training_curves.png", width: 100%),
  caption: "منحنی‌های loss در طول آموزش"
)

#my_heading("آنالیز عملکرد", level: 3)
#block[
#set par(justify: false)
#figure(
  table(
    columns: 3,
    [جنبه], [مدل سفارشی], [YOLOv11L],
    [دقت تشخیص], [متوسط], [بالا],
    [سرعت inference], [متوسط], [سریع],
    [تعداد پارامترها], [~۲۶M], [~۵۹M],
    [اندازه مدل], [کوچک], [بزرگ],
    [قابلیت تنظیم], [بالا], [پایین]
  ),
  caption: [مقایسه مدل سفارشی با YOLOv11L]
)
]

#pagebreak()
#my_heading("ارزیابی بصری نتایج", level: 2)
نتایج حاصل از اعمال مدل سفارشی بر روی تصاویر تست:

#figure(
  image("./cat_and_dog_custom.png", width: 100%),
  caption: "نتیجه مدل سفارشی روی عکس cat_and_dog"
)

#figure(
  image("./random_custom.png", width: 100%),
  caption: "نتیجه مدل سفارشی روی عکس random"
)

#figure(
  image("./two_sample_images_custom.png", width: 100%),
  caption: "نتیجه مدل سفارشی روی دو عکس نمونه از مجموعه داده"
)

#my_heading("تحلیل محدودیت‌ها و بهبودهای ممکن", level: 2)
مدل سفارشی با وجود پیاده‌سازی مناسب، محدودیت‌هایی نسبت به مدل‌های پیشرفته‌تر دارد:

#my_heading("محدودیت‌های فعلی", level: 3)
+ *تعداد anchor box محدود*: استفاده از ۵ anchor box که ممکن است برای اشکال متنوع کافی نباشد
+ *معماری ساده*: نبود feature pyramid یا multi-scale detection
+ *تعداد کلاس محدود*: فقط دو کلاس سگ و گربه
+ *عدم استفاده از تکنیک‌های پیشرفته*: مانند attention mechanisms یا transformer blocks

#my_heading("راه‌های بهبود", level: 3)
+ افزایش تعداد anchor box ها با تحلیل دقیق‌تر داده‌ها
+ پیاده‌سازی Feature Pyramid Network (FPN) برای multi-scale detection
+ استفاده از data augmentation پیشرفته‌تر
+ بهینه‌سازی تابع loss با وزن‌دهی هوشمندتر
+ اعمال تکنیک‌های regularization اضافی

#my_heading("نتیجه‌گیری", level: 2)
در این پروژه، دو روش مختلف برای تشخیص سگ و گربه پیاده‌سازی شد که مدل سفارشی عملکرد بدتری را نسبت به مدل YOLOv11L به نمایش گزاشت


#align(center)[
    #box(
        width: 80%,
        height: auto,
        fill: rgb(240, 240, 255),
        radius: 5pt,
        stroke: 2pt + rgb(100, 100, 255),
        inset: 20pt,
    )[
        #par(justify: false)[
            #text(size: 16pt, weight: "bold")[لینک مخزن گیتهاب]
        ]
        #v(10pt)
        #link("https://github.com/pooyaht/dl_hw2/")[
          Link
        ]
    ]
]