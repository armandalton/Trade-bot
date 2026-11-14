# ربات سیگنال تطبیقی XT — نسخه آزمایشی

این پروژه فقط داده عمومی فیوچرز USDT-M صرافی XT را می‌خواند و در صورت تأیید شرایط، سیگنال را به تلگرام می‌فرستد. **هیچ سفارشی ثبت نمی‌کند و به API Key صرافی نیاز ندارد.**

## مهم قبل از استفاده
- این نسخه MVP و آزمایشی است؛ سودآوری آن ثابت نشده است.
- ابتدا فقط در حالت آزمایشی اجرا شود. سیگنال‌ها را دستی معامله نکنید تا بک‌تست و حداقل چند هفته paper-trading بررسی شود.
- مدل با داده‌های اخیر پارامترها را می‌آزماید؛ کد خودش را تغییر نمی‌دهد.
- بک‌تست ساده است و هزینه فاندینگ/اثر بازار و کیفیت پرشدن سفارش را کامل مدل نمی‌کند؛ نتیجه مثبت به معنی تضمین سود نیست.
- GitHub Actions برای کار دقیق و بی‌وقفه مناسب معامله سریع نیست؛ زمان اجرای schedule ممکن است عقب بیفتد.

## ساخت ربات تلگرام
1. در تلگرام به `@BotFather` پیام بده.
2. `/newbot` را بفرست و نام و username بساز.
3. توکن را محرمانه نگه دار؛ در چت یا کد عمومی قرار نده.
4. ربات جدیدت را باز کن و `/start` بزن.
5. در Safari این آدرس را باز کن و `TOKEN` را فقط روی گوشی خودت جایگزین کن:
   `https://api.telegram.org/botTOKEN/getUpdates`
6. عدد `message.chat.id` را یادداشت کن. اگر چیزی نشان نداد، یک پیام دیگر به ربات بده و صفحه را refresh کن.

## ساخت GitHub Repository با آیفون
1. در github.com وارد حساب شو و `+` → `New repository`.
2. نام: `xt-adaptive-signal-bot`
3. روی `Private` بگذار و `Create repository` را بزن.
4. فایل ZIP را در Files آیفون باز کن و Extract بزن.
5. در Repository روی `Add file` → `Upload files` بزن.
6. محتویات پوشه استخراج‌شده را بارگذاری کن؛ باید `.github/workflows/bot.yml` هم همراه بقیه فایل‌ها باشد.
7. `Commit changes` را بزن.

اگر آپلود پوشه `.github` با آیفون ممکن نبود:
- در Repository روی `Add file` → `Create new file` بزن.
- نام فایل را دقیقاً `.github/workflows/bot.yml` بگذار.
- محتوای فایل `workflow_copy.txt` را کپی و ذخیره کن.
- سپس `bot.py`, `requirements.txt`, `.gitignore` و `README_FA.md` را هم در ریشه Repository آپلود کن.
- پوشه `data` لازم نیست از ابتدا فایل داشته باشد؛ برنامه خودش فایل‌هایش را می‌سازد.

## افزودن Secretهای تلگرام
در Repository:
`Settings` → `Secrets and variables` → `Actions` → `New repository secret`

دو مورد بساز:
- Name: `TELEGRAM_BOT_TOKEN` / Secret: توکن BotFather
- Name: `TELEGRAM_CHAT_ID` / Secret: شناسه عددی چت

توکن را در فایل‌ها یا داخل پیام‌ها ننویس.

## تست اتصال
1. تب `Actions` را باز کن.
2. workflow با نام `XT Adaptive Signal Bot` را انتخاب کن.
3. `Run workflow` را بزن.
4. mode را روی `check` بگذار و اجرا کن.
5. اگر سبز شد، لاگ را باز کن و دنبال `XT connection OK` بگرد. پیام تست هم باید در تلگرام برسد.

## آموزش و ارزیابی
1. دوباره `Run workflow` → mode=`train`.
2. آموزش تاریخی ممکن است چند دقیقه طول بکشد.
3. در پایان فایل `data/model.json` را در GitHub باز کن.
4. فقط اگر `approved` برابر `true` شد، اجرای scan مجاز است. اگر false بود، هیچ سیگنالی صادر نمی‌شود؛ آن را دستی true نکن.

## اجرای اسکن خودکار
Workflow هر ۱۵ دقیقه حالت scan را اجرا می‌کند و روزانه ساعت ۰۰:۱۷ UTC آموزش/ارزیابی مجدد را انجام می‌دهد. اجرای زمان‌بندی‌شده ممکن است با تأخیر باشد.
برای اجرای دستی: `Actions` → workflow → `Run workflow` → mode=`scan`.

## تغییر نمادها یا ریسک
در فایل workflow می‌توان SYMBOLS را تغییر داد. مقادیر پیش‌فرض:
`btc_usdt,eth_usdt,sol_usdt,bnb_usdt,xrp_usdt`

مقادیر فرضی ریسک:
- سرمایه کاغذی: 2000 USDT
- ریسک هر سیگنال: 0.5% = 10 USDT
- حد زیان روزانه: 1.5% = 30 USDT
- افت سرمایه کل: 6% = 120 USDT

توجه: در این MVP، این اعداد فقط چارچوب paper-trading و کنترل سیگنال هستند؛ ربات موجودی حساب واقعی تو را نمی‌خواند و نمی‌تواند زیان واقعی معاملات دستی را تشخیص دهد.

## عیب‌یابی
- `invalid_symbol`: نماد را از لیست قراردادهای قابل معامله XT بررسی کن.
- `Too few candles` یا `No candle data`: ممکن است API محدودیت داده یا فرمت پاسخ عوض شده باشد؛ لاگ Actions را بفرست.
- تلگرام پیام نمی‌دهد: Secretها را دقیق بررسی کن، سپس به ربات `/start` بفرست.
- workflow دیده نمی‌شود: مطمئن شو فایل دقیقاً در `.github/workflows/bot.yml` قرار دارد و روی default branch commit شده است.
- خطای push: در Settings → Actions → General، بخش Workflow permissions را روی `Read and write permissions` بگذار.

## منابع رسمی
- XT API docs: https://doc.xt.com/
- XT Futures kline endpoint: https://doc.xt.com/docs/futures/MarketData/get-trading-pair-information-of-kline
- GitHub Actions schedule: https://docs.github.com/en/actions
