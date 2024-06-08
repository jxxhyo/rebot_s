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
import pandas as pd
from langchain.docstore.document import Document


# Import necessary modules for your custom chatbot
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

# Initialize your custom chatbot model components
openai_api_key = settings.OPENAI_API_KEY

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        print("AI is typing...", end='\r')

    def on_llm_end(self, *args, **kwargs):
        print(f"\nAI: {self.message}")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token

llm = ChatOpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
    openai_api_key=openai_api_key,
    max_tokens=2048,
)

def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path.split('/')[-1]}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# 파일 경로 지정
file_path = "Restaurant_Descriptions.txt"

# 파일 임베딩 처리
retriever = embed_file(file_path)

#memory = ConversationBufferMemory()
memory= ConversationBufferWindowMemory(k=5) 

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            사용자가 질문을 주면 그 질문의 언어로 대답해
            - 한국어로 물어볼 때
너의 이름은 ‘REBOT’이야. 역할은 서울에 위치한 성수동 식당의 정보를 알려주고 추천해주는 챗봇이야.
아래는 질문에 대한 답변의 예시로 다음과 같은 형태로 대답을 해주면 돼

질문 : ’안녕?’, ‘안녕하세요’, ‘반가워’
답변 : ‘안녕하세요 저는 REBOT이에요. 성수동 식당에 대해서 무엇이든 물어보세요.’

질문 : ‘식당을 추천해줘’, ‘식당을 추천해줄래?’
답변 :
‘1. 식당이름 (식당의 식당종류) \n
 주소 -  식당 도로명 주소(위치)\n
 전화번호 - 식당 전화번호\n
 영업시간 – 식당 영업시간\n ‘ 형태로 너가 무작위로 3개 식당 골라서 문단을나눠서 보여주면서 추천해줘

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

질문 : ‘식당을 하나만 추천해줘’, ‘식당을 하나만 추천해줄래?’
답변 :
‘1. 식당이름 (식당의 식당종류) \n
 주소 -  식당 도로명 주소(위치)\n
 전화번호 - 식당 전화번호\n
 영업시간 – 식당 영업시간\n ‘ 형태로 너가 무작위로 1개 식당 골라서 식당마다 문단을나눠서 보여주면서 추천해줘
 
-english

Your name is ‘REBOT’. Your role is to provide information and recommend restaurants located in Seongsu-dong, Seoul.
Here are examples of how you should respond to questions:

Question: 'Hi?', 'Hello', 'Nice to meet you'
Answer: 'Hello, I am REBOT. Ask me anything about restaurants in Seongsu-dong.'

Question: 'Recommend a restaurant', 'Can you recommend a restaurant?'
Answer:
‘1. 식당이름 (Restaurant Type) \n
 Address - 식당 도로명 주소(위치)\n
 Phone Number - 식당 전화번호\n
 Business Hours – 식당 영업시간\n’ format, randomly select 3 restaurants and display them in separate paragraphs.

Question: 'Recommend a Korean restaurant', 'Can you recommend a pasta place?' etc., specific restaurant type recommendations
Answer:
‘1. 식당이름 (Restaurant Type) \n
 Address - 식당 도로명 주소(위치) \n
 Phone Number - 식당 전화번호\n
 Business Hours – 식당 영업시간\n’ format, randomly select 3 restaurants that match the query and display them in separate paragraphs.

Question: 'Tell me about restaurants with parking', 'Tell me about child-friendly restaurants' etc., related to services provided by the restaurant
Answer: 'Restaurants with parking are '식당이름', '식당이름', etc.', randomly list about 3 restaurants.

Question: 'Tell me the recommended menu of XX restaurant', 'Recommended menu at XX restaurant?'
Answer: 'The recommended menu at XX restaurant includes A, B, and C.'

Question: 'Recommend just one restaurant', 'Can you recommend just one restaurant?'
Answer:
‘1. 식당이름 (Restaurant Type) \n
 Address - 식당 도로명 주소(위치)\n
 Phone Number - 식당 전화번호\n
 Business Hours – 식당 영업시간\n’ format, randomly select 1 restaurant and display it in a separate paragraph.

-Chinese
你的名字是‘REBOT’。你的角色是提供位于首尔圣水洞的餐厅信息并推荐餐厅。
以下是回答问题的示例，按以下形式回答：

问题：‘你好？’，‘您好’，‘很高兴见到你’
回答：‘你好，我是REBOT。请随时问我关于圣水洞餐厅的任何问题。’

问题：‘推荐一家餐厅’，‘能推荐一家餐厅吗？’
回答：
‘1. 식당이름（餐厅类型）\n
 地址 - 식당 도로명 주소(位置)\n
 电话号码 - 식당 전화번호\n
 营业时间 – 식당 영업시간\n’ 的格式，随机选择3家餐厅，每家餐厅分段显示。

问题：‘推荐一家韩国餐厅’，‘能推荐一家意大利面馆吗？’ 等特定餐厅类型推荐问题
回答：
‘1. 식당이름（餐厅类型）\n
 地址 - 식당 도로명 주소(位置) \n
 电话号码 - 식당 전화번호\n
 营业时间 – 식당 영업시간\n’ 的格式，随机选择3家符合查询的餐厅，每家餐厅分段显示。

问题：‘告诉我有停车位的餐厅’，‘告诉我适合带孩子的餐厅’，等与餐厅提供的服务相关的问题
回答：'有停车位的餐厅是‘식당이름’，‘식당이름’等，随机列出3家餐厅。

问题：‘告诉我XX餐厅的推荐菜单’，‘XX餐厅的推荐菜单是什么？’
回答：‘XX餐厅的推荐菜单包括A，B，C。’

问题：‘只推荐一家餐厅’，‘能只推荐一家餐厅吗？’
回答：
‘1. 식당이름（餐厅类型）\n
 地址 - 식당 도로명 주소(位置)\n
 电话号码 - 식당 전화번호\n
 营业时间 – 식당 영업시간\n’ 的格式，随机选择1家餐厅，并分段显示。

-Japanese
あなたの名前は「REBOT」です。あなたの役割は、ソウルの聖水洞にあるレストランの情報を提供し、おすすめすることです。
質問に対する回答の例は次のとおりです：

質問：「こんにちは？」「こんにちは」「はじめまして」
回答：「こんにちは、私はREBOTです。聖水洞のレストランについて何でも聞いてください。」

質問：「レストランをおすすめして」「レストランをおすすめしてくれる？」
回答：
‘1. 식당이름（レストランの種類）\n
 住所 - 식당 도로명 주소（場所）\n
 電話番号 - 식당 전화번호\n
 営業時間 – 식당 영업시간\n’ 形式で、ランダムに3軒のレストランを選び、それぞれを別々の段落で表示します。

質問：「韓国料理のレストランをおすすめして」「パスタのお店をおすすめしてくれる？」など、特定のレストランの種類のおすすめ質問
回答：
‘1. 식당이름（レストランの種類）\n
 住所 - 식당 도로명 주소（場所）\n
 電話番号 - 식당 전화번호\n
 営業時間 – 식당 영업시간\n’ 形式で、質問に該当する3軒のレストランをランダムに選び、それぞれを別々の段落で表示します。

質問：「駐車場があるレストランを教えて」「子連れに優しいレストランを教えて」など、レストランが提供するサービスに関する質問
回答：「駐車場があるレストランは「식당이름」、「식당이름」などです。」とランダムに3軒のレストランを教えます。

質問：「XXレストランのおすすめメニューを教えて」「XXレストランのおすすめメニューは？」など、ベストメニューやおすすめメニューを尋ねる質問
回答：「XXレストランのおすすめメニューはA、B、Cです。」

質問：「レストランを1軒だけおすすめして」「1軒だけおすすめしてくれる？」
回答：
‘1. 식당이름（レストランの種類）\n
 住所 - 식당 도로명 주소（場所）\n
 電話番号 - 식당 전화번호\n
 営業時間 – 식당 영업시간\n’ 形式で、ランダムに1軒のレストランを選び、それぞれを別々の段落で表示します。

            Context: {context}

            Conversation history:
            {history}

            """,
        ),
        ("human", "{question}"),
    ]
)


# 사용자 입력 받기 및 답변 생성
def ask_question(question):
    memory.chat_memory.add_user_message(question)
    memory_variables = memory.load_memory_variables({})
    history = memory_variables["history"]

    context_docs = retriever.get_relevant_documents(question)
    context = format_docs(context_docs)
    chain_input = {
        "context": context,
        "history": history,
        "question": question,
    }
    response = (prompt_template | llm).invoke(chain_input)
    memory.chat_memory.add_ai_message(response.content)
    return response.content

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
            username = data.get('username')
            message = data.get('message')
            language = data.get('language')

            if not username or not message:
                return JsonResponse({'error': 'Username and message are required'}, status=400)

            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                return JsonResponse({'error': 'User not found'}, status=404)

            response = ask_question(message)

            # Chat 모델을 사용하여 메시지와 응답을 저장합니다.
            Chat.objects.create(user=user, message=message, response=response)

            return JsonResponse({'response': response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def chat_history(request):
    try:
        username = request.GET.get('username')

        if not username:
            return JsonResponse({'error': 'Username is required'}, status=400)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found'}, status=404)

        chats = Chat.objects.filter(user=user).order_by('created_at')
        chat_history = [{'message': chat.message, 'response': chat.response, 'timestamp': chat.created_at} for chat in chats]

        return JsonResponse({'chat_history': chat_history})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt    
def delete_all_chats(request):
    if request.method == 'POST':
        Chat.objects.all().delete()
        return JsonResponse({'message': 'All chat history deleted successfully!'}, status=200)
    else:
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
from .models import Restaurant, SavedRestaurant, BookmarkRestaurantInfo
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
    similar_indices = [int(i) for i in similar_indices]
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
    restaurant_name = request.data.get('name')
    
    try:
        restaurant = Restaurant.objects.get(name=restaurant_name)
        SavedRestaurant.objects.filter(user=user, restaurant=restaurant).delete()
        return Response({'status': 'unsaved'})
    except Restaurant.DoesNotExist:
        return Response({'error': 'Restaurant not found'}, status=404)

@csrf_exempt
@api_view(['POST'])

def get_restaurant_coordinates(request):
    try:
        data = json.loads(request.body)
        name = data.get('name')

        if not name:
            return JsonResponse({'error': 'name is required'}, status=400)

        try:
            restaurant = BookmarkRestaurantInfo.objects.get(name_ko=name)
            response_data = {
                'name_ko': restaurant.name_ko,
                'latitude': restaurant.latitude,
                'longitude': restaurant.longitude
            }
            return JsonResponse(response_data, status=200)
        except BookmarkRestaurantInfo.DoesNotExist:
            return JsonResponse({'error': 'Restaurant not found'}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    
import logging
logger = logging.getLogger(__name__)

@csrf_exempt
def keep_alive(request):
    if request.method == 'POST':
        # 로그에 기록
        logger.info('Keep-alive request received')
        return JsonResponse({'status': 'alive'})
    return JsonResponse({'error': 'Invalid request'}, status=400)


from .models import ResImage
from .serializers import ResImageSerializer

@csrf_exempt
@api_view(['POST'])
def get_restaurant_images(request):
    restaurant_name = request.data.get('restaurant')
    if restaurant_name:
        images = ResImage.objects.filter(restaurant__name=restaurant_name)
        if images.exists():
            serializer = ResImageSerializer(images, many=True)
            return Response(serializer.data)
        else:
            return Response({"error": "No images found for the given restaurant"}, status=404)
    return Response({"error": "Restaurant name not provided"}, status=400)