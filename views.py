from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import EmailMessage, send_mail
from HY import settings
from django.views.decorators.http import require_POST
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode,urlsafe_base64_decode
from django.utils.encoding import force_bytes,force_str
from django.contrib.auth import authenticate, login, logout
from . tokens import generate_token
from keras.models import load_model
from django.views.decorators.http import require_POST
import json
from django.shortcuts import render
import nltk
from django.http import JsonResponse
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
from nltk.tokenize import sent_tokenize
import json
import random
nltk.download('punkt')
nltk.download('wordnet')

nltk.download('popular')

def home(request):
    return render(request,"auth/index.html")
def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        # gender = request.post['gender']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')
        
        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')
        
        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('home')
        
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        # myuser.is_active = False
        myuser.is_active = False
        myuser.save()
        messages.success(request, "Your Account has been created succesfully!! Please check your email to confirm your email address in order to activate your account.")
         # Welcome Email
        subject = "Welcome to Chatbot Login!!"
        message = "Hello " + myuser.first_name + "!! \n" + "Welcome to Chatbot!! \nThank you for visiting our website\n. We have also sent you a confirmation email, please confirm your email address. \n\nThanking You\nAshu Pabreja"        
        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
        # Email Address Confirmation Email
        current_site = get_current_site(request)
        email_subject = "Confirm your Email @ Chatbot - Login!!"
        message2 = render_to_string('email_confirmation.html',{
            
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser)
        })
        email = EmailMessage(
        email_subject,
        message2,
        settings.EMAIL_HOST_USER,
        [myuser.email],
        )
        email.fail_silently = True
        email.send()
        return redirect('signin')
    return render(request,"auth/signup.html")

def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username=username, password=pass1)
        
        if user is not None:
            login(request, user)
            fname = user.first_name
            # messages.success(request, "Logged In Sucessfully!!")
            return render(request, "auth/index.html",{"fname":fname})
        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('home')
    
    return render(request, "auth/signin.html")

def activate(request,uidb64,token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser,token):
        myuser.is_active = True
        # user.profile.signup_confirmation = True
        myuser.save()
        login(request,myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('signin')
    else:
        return render(request,'activation_failed.html')  
def signout(request):
   logout(request)
   messages.success(request, "Logged Out Successfully!!")
   return redirect('home')

lemmatizer = WordNetLemmatizer()

model = load_model('finalmodel.h5')
intents = json.loads(open('Final_2511.json').read())
words = pickle.load(open('final_texts.pkl', 'rb'))
classes = pickle.load(open('final_labels.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
wrds = []
ignore_letters = ['?', '!']
wrds = [lemmatizer.lemmatize(w) for w in wrds if w not in ignore_letters]
wrds = sorted(set(wrds))
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))

def predict_class(sentence,model):


    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            responses = i['responses']
            response = random.choice(responses)
            sentences = sent_tokenize(response)
            sentence = random.choice(sentences)
            break
    return sentence

# @csrf_exempt
@csrf_exempt
# @require_POST
def chatbot_view(request):
    if request.method == 'POST':
        user_message = json.loads(request.body.decode('utf-8')).get('message', '')
        if user_message.lower() == 'exit':
            return JsonResponse({'response': 'Chatbot Response: Have a nice day!'})

        ints = predict_class(user_message, model)
        chatbot_response = get_response(ints, intents)
        
        return JsonResponse({'response':  chatbot_response})

    
    return render(request, 'auth/chatbot_view.html')
from django.shortcuts import render

def feedback(request):
    return render(request, 'auth/feedback.html')
  