from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.models import User
from .models import Chat, Profile
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
import json
from django.contrib.auth import login, logout
from django.conf import settings

# Import necessary modules for your custom chatbot
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize your custom chatbot model components
openai_api_key = settings.OPENAI_API_KEY

# List of CSV file paths
csv_files = ['restaurant_info1.csv','all_res_info_df.csv','animal.csv']

# Load documents from all CSV files
documents = []
for file_path in csv_files:
    loader = CSVLoader(file_path=file_path)
    documents.extend(loader.load_and_split())

# Initialize embeddings
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize vector store with all documents
vectorstore = Chroma.from_documents(documents=documents, embedding=hf_embeddings)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

system_template = """너의 이름은 ‘REBOT’이야. 역할은 서울에 위치한 성수동 식당의 정보를 알려주고 추천해주는 챗봇이야. 사용자가 물어본 질문에 대해서 다음과 같이 대답하면 됩니다.
잘 모르는 내용이면 ‘해당 내용에 대해서는 잘 모르겠습니다.’라고 대답합니다.

질문 : ’안녕?’, ‘안녕하세요’, ‘반가워’ 
답변 : ‘안녕하세요 저는 REBOT이에요. 성수동 식당에 대해서 무엇이든 물어보세요.’

질문 : ‘식당을 추천해줘’, ‘식당을 추천해줄래?’ 	
답변 : 
‘1. 식당이름 (식당의 식당종류) \n
 주소 -  식당 도로명 주소(위치)\n
 전화번호 - 식당 전화번호\n
 영업시간 – 식당 영업시간\n ‘ 형태로 너가 무작위로 5개 식당 골라서 식당마다 문단을나눠서 보여주면서 추천해줘

질문 : ‘한식당 식당을 추천해줘’, ‘파스타집 추천해줄래?’ 등 특정 식당 종류 추천 질문	
답변 : 
‘1. 식당이름 (식당의 식당종류) \n
 주소 -  식당 도로명 주소(위치) \n
 전화번호 - 식당 전화번호\n
 영업시간 – 식당 영업시간\n ‘ 형태로 너가 무작위로 질문에 해당되는 식당 3개만 골라서 식당마다 문단을 나눠서 보여주면서 추천해줘

 질문 : '주차 가능한 식당 알려줘','유아동반 가능한 식당 알려줘' 등 식당에서 제공하는 서비스와 관련된 질문이 들어올 경우 
 답변 : '주차가 가능한 식당은 '식당이름','식당이름'입니다.'와 같은 형식으로 랜덤으로 뽑아서 식당 3개 정도만 알려줘.
 
질문 : 'XX식당 추천메뉴 알려줘' 'XX식당 추천메뉴?' 등 베스트메뉴나 추천메뉴를 물어보는 질문
답변 : 'XX식당의 추천메뉴는 A,B,C 입니다.'라는 형식으로 답변해.

 ---------------- {context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# Initialize LLM and retriever
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, max_tokens=2048, model_name='gpt-4o', streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
retriever = vectorstore.as_retriever()

# Initialize the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, combine_docs_chain_kwargs={"prompt": qa_prompt}, memory=memory, output_key='answer')

def ask_openai(message):
    response = qa({"question": message})
    answer = response['answer']
    return answer

@csrf_exempt
def register(request):
    try:
        data = json.loads(request.body)
        username = data.get('username')
        email = data.get('email')
        password1 = data.get('password1')
        password2 = data.get('password2')
        language = data.get('language')

        if password1 != password2:
            return JsonResponse({'error': "Passwords don't match"}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': "Username already exists"}, status=400)

        if User.objects.filter(email=email).exists():
            return JsonResponse({'error': "Email already exists"}, status=400)

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()

        # Profile 생성은 신호 수신자가 처리하므로, 언어를 설정하는 부분만 추가합니다.
        profile = user.profile  # get_or_create_profile 메서드 사용
        profile.language = language
        profile.save()

        login(request, user)
        return JsonResponse({'message': 'User registered successfully!'}, status=200)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate, login

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        password = data.get('password')

        print(f"Email: {email}, Password: {password}")  # 디버깅 메시지 추가

        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)

            # JWT 토큰 생성
            refresh = RefreshToken.for_user(user)
            response = JsonResponse({
                'success': True,
                'message': 'Login successful',
                'username': user.username,
                'access_token': str(refresh.access_token),
                'refresh_token': str(refresh),
            })
            response.set_cookie('access_token', str(refresh.access_token), httponly=True)
            response.set_cookie('refresh_token', str(refresh), httponly=True)
            return response
        else:
            return JsonResponse({'error': 'Invalid email or password'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return JsonResponse({'message': 'Logged out successfully'}, status=200)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            if not message:
                return JsonResponse({'error': 'Message is required'}, status=400)

            response = ask_openai(message)
            return JsonResponse({'response': response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

#@login_required
def get_user_info(request, username):
    try:
        user = User.objects.get(username=username)
        profile = user.profile
        return JsonResponse({
            'username': user.username,
            'language': profile.language
        })
    except User.DoesNotExist:
        return JsonResponse({'error': 'User does not exist'}, status=404)

@ensure_csrf_cookie
def set_csrf_token(request):
    return JsonResponse({'detail': 'CSRF cookie set'})

from rest_framework import viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import Restaurant, SavedRestaurant
from .serializers import RestaurantSerializer, SavedRestaurantSerializer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import int64


class RestaurantViewSet(viewsets.ModelViewSet):
    queryset = Restaurant.objects.all()
    serializer_class = RestaurantSerializer
    permission_classes = [IsAuthenticated]

@api_view(['GET'])
#@permission_classes([IsAuthenticated])
def saved_restaurants(request):
    user = request.user
    saved_restaurants = SavedRestaurant.objects.filter(user=user)
    serializer = SavedRestaurantSerializer(saved_restaurants, many=True)
    return Response(serializer.data)

@api_view(['GET'])
#@permission_classes([IsAuthenticated])
def recommend_restaurants(request):
    user = request.user
    saved_restaurants = SavedRestaurant.objects.filter(user=user).values_list('restaurant', flat=True)
    if not saved_restaurants:
        return Response([])

    saved_restaurant_objs = Restaurant.objects.filter(id__in=saved_restaurants)
    all_restaurants = Restaurant.objects.exclude(id__in=saved_restaurants)
    
    all_data = [
        f"{r.menu1} {r.menu2}  {r.service} {r.category} {r.location}"
        for r in all_restaurants
    ]

    saved_data = [
        f"{r.menu1} {r.menu2} {r.service} {r.category} {r.location}"
        for r in saved_restaurant_objs
    ]

    vectorizer = CountVectorizer().fit_transform(saved_data + all_data)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    saved_indices = range(len(saved_data))
    similar_indices = cosine_matrix[saved_indices, len(saved_data):].mean(axis=0).argsort()[::-1][:5]
    similar_indices = [int(i) for i in similar_indices]  # int64를 int로 변환
    similar_restaurants = [all_restaurants[i] for i in similar_indices]

    serializer = RestaurantSerializer(similar_restaurants, many=True)
    return Response(serializer.data)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Restaurant, SavedRestaurant

@api_view(['POST'])
#@permission_classes([IsAuthenticated])
def save_restaurant(request):
    user = request.user
    restaurant_name = request.data.get('name')

    if not restaurant_name:
        return Response({'error': 'Restaurant name is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        restaurant = Restaurant.objects.get(name=restaurant_name)
    except Restaurant.DoesNotExist:
        return Response({'error': 'Restaurant not found'}, status=status.HTTP_404_NOT_FOUND)

    saved_restaurant, created = SavedRestaurant.objects.get_or_create(user=user, restaurant=restaurant)

    if created:
        return Response({'status': 'saved'}, status=status.HTTP_201_CREATED)
    else:
        return Response({'status': 'already saved'}, status=status.HTTP_200_OK)

@api_view(['POST'])
#@permission_classes([IsAuthenticated])
def unsave_restaurant(request):
    user = request.user
    restaurant_id = request.data.get('id')
    restaurant = Restaurant.objects.get(id=restaurant_id)
    SavedRestaurant.objects.filter(user=user, restaurant=restaurant).delete()
    return Response({'status': 'unsaved'})