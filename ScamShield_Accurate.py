import re
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title='ScamShield AI', page_icon='🛡️', layout='centered')
st.markdown('''<style>
.stApp {background:#f7f5ff;color:#241b43}
.stApp p,.stApp label,.stApp li,.stApp span {color:#241b43}
h1,h2,h3 {color:#43348c!important}
[data-testid="stTextArea"] textarea {background:#ffffff!important;color:#241b43!important;-webkit-text-fill-color:#241b43!important;border:1px solid #a99ce4!important}
[data-testid="stTextArea"] textarea::placeholder {color:#6c6482!important;-webkit-text-fill-color:#6c6482!important}
.stButton button {background:#6554c0!important;color:#ffffff!important;border-radius:12px!important}
[data-testid="stAlert"] p,[data-testid="stAlert"] span {color:inherit!important}
</style>''', unsafe_allow_html=True)

# Educational synthetic examples; no claims of real-world coverage.
scam_templates = [
 'Your bank account will be blocked unless you verify your OTP now',
 'Urgent KYC update needed enter your password at this link',
 'Congratulations you won a prize send your bank details to claim',
 'Your parcel is held pay a small redelivery fee through this link',
 'I am your principal buy gift cards and send me the codes privately',
 'Your electricity supply will be cut today pay through this link immediately',
 'Your account has been suspended click here and enter your PIN',
 'Your tax refund is waiting provide card details to receive it',
 'Your package delivery failed confirm address and pay a rescheduling fee',
 'I am your manager purchase vouchers and keep this request secret',
 'You are selected for a job pay registration charges before joining',
 'Earn guaranteed money every day deposit a joining fee now',
 'Your payment failed update credit card information using this link',
 'Unusual login detected share the verification code to restore access',
 'Your SIM will stop working submit Aadhaar and OTP to reactivate',
 'Your loan is approved pay an advance processing fee first',
 'Claim your lottery winnings by sending account number and password',
 'Police case registered against you transfer money to close it',
 'I am from customer support tell me the code sent to your phone',
 'Your subscription expires today click the unfamiliar link to pay',
 'Aapka bank account block ho jayega OTP bhejo abhi',
 'KYC update karo warna account band ho jayega link pe click karo',
 'Aap lottery jeet gaye bank details bhejo prize claim karne ke liye',
 'Parcel deliver nahi hua redelivery ke liye payment karo',
 'Main principal bol raha hoon gift card kharido aur code privately bhejo',
 'Bijli kat jayegi abhi payment link kholo',
 'Aapka ATM card band ho raha hai PIN share karo',
 'Naukri pakki hai pehle registration fee jama karo',
 'Aapko cashback mila hai OTP batao paise lene ke liye',
 'Mujhe urgent gift voucher chahiye kisi ko mat batana',
 'Aapka package hold par hai address verify karne ke liye paise do',
 'Main bank se bol raha hu password aur OTP share karein',
 'Aapke naam par police complaint hai fine abhi transfer karo',
 'Account freeze hone wala hai is link se details update karein',
 'Free mobile jeeta hai processing charges bhejo',
 'Aapka refund pending hai card number aur CVV bhejo',
]
normal_templates = [
 'Your science class starts at eight tomorrow bring your notebook',
 'Please return your library book before Friday',
 'Your dentist appointment is confirmed for tomorrow at four',
 'Your order shipped track it through the official shopping app',
 'The school assembly will start at nine in the morning',
 'Can we meet for group study after school today',
 'The teacher uploaded the homework notes to the classroom portal',
 'Happy birthday hope you have a wonderful day',
 'Your parcel was delivered to the reception desk',
 'The museum opens at ten and closes at six',
 'Please review the meeting agenda before tomorrow',
 'Our football practice has moved to Thursday',
 'The science exhibition is next week in the school hall',
 'Dinner is ready please come downstairs',
 'The bus will arrive at the usual stop',
 'Reminder to bring your maths textbook on Monday',
 'You can pay your bill in the official app if it is due',
 'Do not share your OTP or password with anyone',
 'Beware of fake prize messages asking for bank details',
 'Our school principal announced the annual day celebration',
 'Kal science ka test hai notebook lekar aana',
 'Aaj library ki book return kar dena',
 'Kal dentist appointment hai chaar baje',
 'Aapka order dispatch ho gaya official app par track karein',
 'School assembly subah nau baje shuru hogi',
 'Homework ki photo class group mein bhej dena',
 'Happy birthday aapka din bahut accha ho',
 'Parcel reception par deliver ho gaya hai',
 'Aaj football practice cancel ho gayi hai',
 'Maths ki extra class kal hogi',
 'Kisi ko bhi apna OTP share mat karna',
 'Fake lottery message se savdhan rahein',
 'School principal ne annual function ki date batayi',
 'Aap official bank app se balance check kar sakte hain',
 'Mujhe gift card mila birthday par thank you',
 'Kal meeting hai please time par aa jana',
]
# Keep an independent set of examples separate from training.
holdout = [
 ('A courier says pay a delivery charge on an unknown link or lose your package',1),
 ('I am the head teacher buy a voucher now and send the code secretly',1),
 ('Aapka account band hoga OTP share karo turant',1),
 ('Job offer confirmed pehle joining fee bhejo',1),
 ('Your refund requires your CVV and card number',1),
 ('Kal school mein science exhibition hai',0),
 ('Your doctor appointment is confirmed for Monday',0),
 ('Do not give anyone your OTP even if they claim to be the bank',0),
 ('Please bring your library books tomorrow',0),
 ('Track your order in the official store app',0),
]

@st.cache_resource
def train_model():
    texts = scam_templates + normal_templates
    labels = [1]*len(scam_templates)+[0]*len(normal_templates)
    model = make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,5),sublinear_tf=True,min_df=1),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42))
    model.fit(texts,labels)
    xs=[m for m,y in holdout]
    ys=[y for m,y in holdout]
    pred=model.predict(xs)
    return model, {'accuracy':accuracy_score(ys,pred),'precision':precision_score(ys,pred,zero_division=0),'recall':recall_score(ys,pred,zero_division=0),'cm':confusion_matrix(ys,pred,labels=[0,1]).tolist()}

model,metrics=train_model()

# Context-sensitive red flags: phrases requesting secrets, money or unsafe action.
RULES = [
 ('Requests an OTP, PIN, password or card security code', r'(?:share|send|enter|provide|tell|batao|bhejo|batana|share karo|submit|daalo|dena|de do).{0,65}\b(?:otp|pin|password|cvv|verification code|security code)\b|\b(?:otp|pin|password|cvv|verification code|security code)\b.{0,65}(?:share|send|enter|provide|tell|batao|bhejo|submit|daalo|dena|de do)'),
 ('Requests a gift card or voucher code', r'(?:buy|purchase|kharido|send|bhejo).{0,60}(?:gift card|voucher)|(?:gift card|voucher).{0,60}(?:code|send|bhejo|privately)'),
 ('Demands a fee to claim a prize, job or refund', r'(?:prize|lottery|job|naukri|refund|cashback|reward).{0,100}(?:fee|charge|payment|deposit|paise|processing)|(?:fee|charge|payment|deposit|paise).{0,100}(?:prize|lottery|job|naukri|refund|reward)'),
 ('Threatens account closure or service disconnection', r'(?:account|bank|sim|electricity|bijli|kyc).{0,65}(?:blocked|suspended|freeze|frozen|band ho|cut|disconnect|kat jayegi)|(?:blocked|suspended|freeze|frozen).{0,65}(?:account|bank)'),
 ('Requests a delivery or redelivery payment', r'(?:parcel|package|delivery|courier|redelivery).{0,100}(?:fee|pay|payment|charge|paise|₹|rs\.?\s*\d+)'),
 ('Asks for secrecy around a request', r'(?:don.t tell anyone|keep (?:this|it) secret|send .*privately|kisi ko mat batana|secret rakhna)'),
 ('Pushes an unfamiliar link or urgent verification', r'(?:click|open|visit|link pe|link kholo).{0,55}(?:link|http|verify|payment|details)|(?:verify|update).{0,55}(?:http|www\.)'),
]
NEGATION = re.compile(r'\b(?:do not|don.t|never|avoid|beware|warning|scam alert|mat|nahi|nahin|savdhan)\b',re.I)

def analyze(message):
    ml_score=float(model.predict_proba([message])[0][1])
    flags=[name for name,pattern in RULES if re.search(pattern,message,re.I)]
    # Safety education messages must not be treated as requests simply because they mention scam vocabulary.
    warning_context=bool(NEGATION.search(message)) and bool(re.search(r'\b(?:otp|password|scam|fraud|lottery|pin)\b',message,re.I))
    if warning_context and not re.search(r'\b(?:i am|main .* bol raha|your .* will|aapka .* hoga)\b',message,re.I):
        flags=[]
    if warning_context and not flags: ml_score=min(ml_score,0.20)
    if len(flags)>=1:
        status='⚠️ Treat this request as suspicious'
        severity='warning'
    elif ml_score>=0.65:
        status='⚠️ Scam-like language detected'
        severity='warning'
    elif ml_score>=0.40:
        status='⚠️ Uncertain — verify independently'
        severity='caution'
    else:
        status='Fewer scam-like patterns detected'
        severity='low'
    return ml_score,flags,status,severity

st.title('🛡️ ScamShield AI')
st.caption('English + Hinglish • Student cybersecurity awareness project')
st.write('Analyze a fictional message for scam-like language and understand possible warning signs. This tool is educational, not a definitive scam detector.')
examples={
    'Write my own message':'',
    'Fake bank KYC':'Aapka bank account block ho jayega. OTP bhejo abhi!',
    'Fake principal':'I am your school principal. Buy a gift card and send me the code privately. Do not tell anyone.',
    'Suspicious delivery':'Your parcel cannot be delivered. Pay ₹25 redelivery fee using this link.',
    'Normal reminder':'Kal science ka test hai. Notebook lekar aana.',
    'Safety advice':'Do not share your OTP or password with anyone.',
}
choice=st.selectbox('Try an example (optional)',list(examples))
message=st.text_area('Message to check',value=examples[choice],height=145,placeholder='Type an example here. Never paste real OTPs, passwords or personal information.')
if st.button('🔎 Analyze Message',type='primary',use_container_width=True):
    if not message.strip():
        st.warning('Please enter a message first.')
    else:
        score,flags,status,severity=analyze(message)
        st.subheader('Analysis')
        if severity=='warning': st.warning(status)
        elif severity=='caution': st.info(status)
        else: st.success(status)
        st.progress(score,text=f'Machine-learning scam-like score: {score:.0%} (not a real-world scam probability)')
        st.write('**Detected warning signs:**')
        if flags:
            for flag in flags: st.write('• '+flag)
        else:
            st.write('No specific high-risk request patterns found. This does not prove a message is safe.')
        st.info('Never share OTPs, passwords, PINs or banking details. Verify unexpected requests using an official app, website or phone number you already trust.')
        st.caption('The ML score and safety warnings are separate. Strong rule-based warnings can appear even when the model score is moderate.')

with st.expander('How the AI works and how it was tested'):
    st.write('A character n-gram TF-IDF vectorizer and logistic-regression classifier learn patterns from synthetic English and Hinglish examples. Separate rules flag requests for sensitive data, unusual payments, secrecy and suspicious links.')
    st.write('**Automated holdout evaluation (10 synthetic examples excluded from training):**')
    st.write(f"Accuracy: {metrics['accuracy']:.0%} · Scam precision: {metrics['precision']:.0%} · Scam recall: {metrics['recall']:.0%}")
    st.caption('This is a very small, synthetic test set. Its score must NOT be presented as real-world accuracy. New wording, languages and genuine urgent requests can cause errors.')
    st.write('Confusion matrix [[true normal, false alarm], [missed scam, detected scam]]:')
    st.code(str(metrics['cm']))
    st.write('Known limitations: mixed Hindi scripts, novel scam strategies, social context and legitimate payment requests are not fully covered. Human verification is essential.')
