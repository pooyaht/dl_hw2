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
معماری مدل شامل دو قسمت اصلی می‌باشد:

#my_heading("شبکه پشتیبان (Backbone)", level: 3)
از مدل ResNet50 پیش‌آموزش‌دیده بر روی مجموعه‌داده ImageNet به عنوان شبکه پشتیبان استفاده شده است. برای سازگاری با معماری YOLO، لایه‌های fully connected انتهایی(دو لایه آخر) حذف شده و فقط قسمت convolutional باقی مانده است. در این مدل از feature map هایی با سایز ۷*۷ و ۱۴*۱۴ پشتیبانی می‌شود.

#my_heading("سر پیش‌بینی (Prediction Head)", level: 3)
سر پیش‌بینی شامل چهار لایه convolutional است:
+ لایه اول: کاهش کانال‌ها از ۲۰۴۸ به ۲۵۶
+ لایه دوم: convolution با kernel سایز ۳ و dilation برابر ۲
+ لایه سوم: convolution با kernel سایز ۱ برای ترکیب feature ها
+ لایه آخر: تولید خروجی نهایی

هر لایه (به جز آخری) دارای Batch Normalization، ReLU و Dropout است.

خروجی نهایی برای هر anchor box شامل:
- ۴ مختصات bounding box (x, y, w, h)
- ۱ امتیاز اطمینان (objectness score)
- ۲ امتیاز کلاس (سگ و گربه)

به طور مثال اگر اندازه feature map را ۷*۷ و تعداد anchor ها را ۳ در نظر بگیریم،\ $۷*۷*۳*(۴+۲+۱)$ پیش‌بینی خواهیم داشت.

#my_heading("استراتژی آموزش", level: 3)
برای بهینه‌سازی فرآیند آموزش، از استراتژی "Progressive Unfreezing" استفاده شده است:

+  تا epoch ۱۰: تمام لایه‌های backbone منجمد هستند
+  از epoch ۱۰ تا ۲۰: دو لایه آخر backbone آزاد می‌شوند
+  از epoch ۲۰ به بعد: تمام لایه‌های backbone آزاد می‌شوند

*توجه شود epoch ها توسط ابرپارامتر#footnote[Hyperparameter] تعیین می‌شوند و اعداد بالا صرفا یک مثال می‌باشد*


#my_heading("تنظیمات آموزش", level: 2)
پارامترهای اصلی آموزش به شرح زیر تعیین شده‌اند:

- Grid Size: ۱۴×۱۴
- Anchor Boxes: بر اساس تحلیل ابعاد bounding box های داده‌های آموزش
- Dropout Rate: ۰.۲ برای جلوگیری از overfitting
- Learning Rate: تطبیقی بر اساس وضعیت unfreezing
- Batch Size: بر اساس محدودیت‌های حافظه GPU

#my_heading("Loss Function", level: 3)
تابع هزینه ترکیبی شامل سه بخش اصلی است:
- Localization Loss: برای دقت مختصات bounding box
- Confidence Loss: برای تشخیص وجود object
- Classification Loss: برای تشخیص نوع حیوان (سگ یا گربه)

#my_heading("نتایج و ارزیابی")
نتایج حاصل از اعمال مدل سفارشی بر روی دو تصویر تست:



#my_heading("مقایسه نتایج", level: 2)

#block[
#set par(justify: false)
#figure(
  table(
    columns: 4,
    [مدل], [دقت تشخیص], [سرعت پردازش], [حجم مدل],
    [YOLOv11L], [بالا], [متوسط], [بزرگ],
    [مدل سفارشی], [متوسط], [سریع], [متوسط]
  ),
  caption: [مقایسه کلی عملکرد دو مدل]
)
]

مدل سفارشی با وجود داشتن دقت کمتر نسبت به YOLOv11L، از نظر سرعت پردازش و تطبیق با داده‌های خاص (سگ و گربه) عملکرد مناسبی نشان می‌دهد.

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
            #text(size: 16pt, weight: "bold")[لینک نوت‌بوک مدل سفارشی در Colab ]
        ]
        #v(10pt)
        #link("https://colab.research.google.com/drive/your_custom_model_link")[
            #image("./colab-badge.svg")
        ]
    ]
]

#my_heading("نتیجه‌گیری")
در این پروژه دو رویکرد مختلف برای تشخیص سگ و گربه بررسی شد. مدل YOLOv11L با دقت بالا اما پیچیدگی محاسباتی زیاد، و مدل سفارشی با سرعت بالا و قابلیت تطبیق بیشتر با مسئله خاص. انتخاب بین این دو مدل بستگی به نیازهای خاص پروژه و محدودیت‌های محاسباتی دارد.

برای بهبود عملکرد مدل سفارشی پیشنهادات زیر ارائه می‌شود:
- افزایش تنوع داده‌های آموزش
- تنظیم دقیق‌تر anchor boxes
- استفاده از techniques پیشرفته‌تر augmentation
- بهینه‌سازی hyperparameter های مدل