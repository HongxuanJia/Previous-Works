import json
import random
import datetime
import time
import numpy as np
from django.db.models import Q
from django.http import JsonResponse

# Create your views here.
from django.views.decorators.http import require_http_methods

from gift import models
from identity.account import models as IdentityModels
from .models import Gift
from identity.account.models import Person, Like, Collect, GiftViewHistory


# Search method
def index(request, search_text):
    # Obtain the gift items
    # all_gifts = models.Gift.objects.all()
    # for gift in all_gifts:
    #     print(gift.gift_name)
    # If receiving the POST request consult the corresponding data
    if request.method == 'GET':
        # Error handling
        if search_text == "":
            error = {"error": "Sorry, gift name cannot be empty, try again~"}
            return JsonResponse(error)

        elif not models.Gift.objects.filter(Q(gift_name__icontains=search_text) | Q(gift_intro__icontains=search_text)):
            error = {search_text: "Sorry, gift currently not available~"}
            return JsonResponse(error, status=404)
        else:
            found_gifts = models.Gift.objects.filter(
                Q(gift_name__icontains=search_text) | Q(gift_intro__icontains=search_text))
            found_gifts_json_list = []
            for gift in found_gifts:
                gift_dic = {
                    'gift_id': gift.pk,
                    'gift_name': gift.gift_name,
                    'gift_description': gift.gift_intro,
                    'gift_img': str("http://fooxking.net:9000/media/" + str(gift.gift_img)),
                    'tag_color': gift.tag_color,
                    'tag_festival': gift.tag_festival,
                    'tag_price': gift.tag_price,
                    'tag_user': gift.tag_user,
                    'tag_species': gift.tag_species,
                    'link': gift.link,
                    'num_thumb_up': gift.num_thumb_up,
                    "like_status": False,
                    "star_status": False
                }
                if request.user.is_authenticated:
                    if Like.objects.filter(person=request.user.person, gift_id=gift.pk).exists():
                        gift_dic["like_status"] = True
                    if Collect.objects.filter(person=request.user.person, gift_id=gift.pk).exists():
                        gift_dic["star_status"] = True

                found_gifts_json_list.append(gift_dic)
            return JsonResponse(found_gifts_json_list, safe=False)


# If click on one selected gift, add the corresponding search_number by one
def click_gift(request, gift_id):
    try:
        click_gift = models.Gift.objects.get(pk=gift_id)

    except Exception as e:
        msg = {
            "status": 0,
            "msg": "No such item",
            "error": e
        }
        return JsonResponse(msg, status=404)

    num_search = click_gift.num_search
    # Query all the related images of the found gift
    gift_img_repo = models.gift_imgs_repo.objects.filter(gift__gift_name=click_gift.gift_name)

    # Add all the images into a list
    gift_img_list = []
    if click_gift.gift_img:
        gift_img_list.append(str("http://fooxking.net:9000/media/" + str(click_gift.gift_img)))

    for imgs in gift_img_repo:
        img_addr = str("http://fooxking.net:9000/media/" + str(imgs.more_img))
        gift_img_list.append(img_addr)

    click_gift.num_search = str(int(num_search) + 1)  # add number of lick by one
    click_gift.save()
    gift_dic = {
        'gift_id': click_gift.pk,
        'gift_name': click_gift.gift_name,
        'gift_description': click_gift.gift_intro,
        'gift_img': gift_img_list,
        'tag_color': click_gift.tag_color,
        'tag_festival': click_gift.tag_festival,
        'tag_price': click_gift.tag_price,
        'tag_user': click_gift.tag_user,
        'tag_species': click_gift.tag_species,
        'link': click_gift.link,
        'num_thumb_up': click_gift.num_thumb_up,
        'like_status': False,
        'star_status': False,
    }
    if request.user.is_authenticated:
        if Like.objects.filter(person=request.user.person, gift_id=gift_id).exists():
            gift_dic["like_status"] = True
        if Collect.objects.filter(person=request.user.person, gift_id=gift_id).exists():
            gift_dic["star_status"] = True

        history = GiftViewHistory.objects.create(gift=click_gift, person=request.user.person)
        history.save()

    return JsonResponse(gift_dic, safe=False)


# Return top ten gifts list
def top_ten_gift(request):
    all_gifts = models.Gift.objects.all()
    score_dict = {}
    all_likes = IdentityModels.Like.objects.all()
    all_collections = IdentityModels.Collect.objects.all()

    # compute score for each gift
    for gift in all_gifts:
        gift_score = 0
        gift_like = 0
        gift_collect = 0

        gift_create_time = gift.create_time
        time_now = datetime.datetime.now().date()

        for like_obj in all_likes:
            if gift.gift_name == like_obj.gift.gift_name:
                gift_like += 1

        for collect_obj in all_collections:
            if gift.gift_name == collect_obj.gift.gift_name:
                gift_collect += 1

        delta = time_now - gift_create_time
        gift_score += (gift_collect * 10 + gift_like * 5 + gift.num_search) / \
                      ((delta.days + 3) ^ 2)

        score_dict.update({gift.pk: gift_score})
    sorted_gifts = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_gifts)

    # Add top ten gifts into a new dict
    required_cnt = 10
    cnt = 0
    top_ten_dict = {}
    for key, value in sorted_gifts.items():
        cnt += 1
        if cnt > required_cnt:
            break
        top_ten_dict.update({key: value})
    # print(top_ten_dict)

    # search top ten gift in database and add top ten into a dict list
    top_ten_list = []
    for key in top_ten_dict.keys():
        top_ten_gift = models.Gift.objects.get(pk=key)
        gift_dict = {
            'gift_id': top_ten_gift.pk,
            'gift_name': top_ten_gift.gift_name,
            'gift_description': top_ten_gift.gift_intro,
            'gift_img': str("http://fooxking.net:9000/media/" + str(top_ten_gift.gift_img)),
            'tag_color': top_ten_gift.tag_color,
            'tag_festival': top_ten_gift.tag_festival,
            'tag_price': top_ten_gift.tag_price,
            'tag_user': top_ten_gift.tag_user,
            'tag_species': top_ten_gift.tag_species,
            'link': top_ten_gift.link,
            'num_thumb_up': top_ten_gift.num_thumb_up,
        }
        top_ten_list.append(gift_dict)
    # print(top_ten_list)

    return JsonResponse(top_ten_list, safe=False)


def all_gifts(request):
    all_id_list = []
    all_object = models.Gift.objects.all()
    for item in all_object:
        all_id_list.append(item.pk)

    return JsonResponse(all_id_list, safe=False)


@require_http_methods(["GET"])
def recommend_random(request):
    if not Gift.objects.exists():
        msg = {
            "status": 0,
            "msg": "no gift available"
        }
        return JsonResponse(msg, status=404)

    gift_list = []
    if Gift.objects.count() < 4:
        for gift in Gift.objects.all():
            gift_list = _gift_list_maker(gift_list, gift)
    else:
        source_list = Gift.objects.order_by('?')
        for i in range(4):
            gift = source_list[i]
            gift_list = _gift_list_maker(gift_list, gift)

    msg = {
        "status": 1,
        "gift_list": gift_list
    }
    return JsonResponse(msg, status=200)


def _gift_list_maker(gift_list, gift):
    gift_info = {
        'gift_id': gift.pk,
        'gift_name': gift.gift_name,
        'gift_description': gift.gift_intro,
        'gift_img': "http://fooxking.net:9000/media/" + str(gift.gift_img),
        'tag_color': gift.tag_color,
        'tag_festival': gift.tag_festival,
        'tag_price': gift.tag_price,
        'tag_user': gift.tag_user,
        'tag_species': gift.tag_species,
        'link': gift.link,
        'num_thumb_up': gift.num_thumb_up
    }
    gift_list.append(gift_info)
    return gift_list


def _recommend_random(request):
    def index_freedom():
        count = models.Gift.objects.count()
        # print(count)
        index_list = []
        gift_id_list = []
        gift = models.Gift.objects.all()
        for gift in gift:
            id_gift = gift.pk
            gift_id_list.append(id_gift)

        index_list = random.sample(gift_id_list, 4)
        # print(index_list)
        # while index in index_list:
        #     index = random.randint(gift_id_list)
        # index_list.append(index)
        return index_list

    gift_list = []
    for i in range(0, 4):
        # print(i)
        index_list = index_freedom()
        gift_item = models.Gift.objects.get(pk=index_list[i])
        gift_dict = {
            'gift_id': gift_item.pk,
            'gift_name': gift_item.gift_name,
            'gift_description': gift_item.gift_intro,
            'gift_img': str("http://fooxking.net:9000/media/" + str(gift_item.gift_img)),
            'tag_color': gift_item.tag_color,
            'tag_festival': gift_item.tag_festival,
            'tag_price': gift_item.tag_price,
            'tag_user': gift_item.tag_user,
            'tag_species': gift_item.tag_species,
            'link': gift_item.link,
            'num_thumb_up': gift_item.num_thumb_up
        }
        gift_list.append(gift_dict)
    return JsonResponse(gift_list, safe=False)


def post_list(request):
    post_list_items = models.gift_post.objects.all()
    post_list = []
    return_dict = {
        'status': 1,
        'post_list': post_list
    }

    for post in post_list_items:

        # query all the related images in image repo
        post_img_repo = models.gift_imgs_repo.objects.filter(post__post_title=post.post_title)

        # Add all the images into a list
        post_img_list = []
        if post.post_img:
            post_img_list.append(str("http://fooxking.net:9000/media/" + str(post.post_img)))

        for imgs in post_img_repo:
            post_addr = str("http://fooxking.net:9000/media/" + str(imgs.more_img))
            post_img_list.append(post_addr)

        # print(post.post_title)
        post_dict = {
            'post_id': post.pk,
            'post_title': post.post_title,
            'post_imgs': post_img_list
        }
        post_list.append(post_dict)
    return JsonResponse(return_dict)


# return a single post
def post(request, post_id):
    post = models.gift_post.objects.get(pk=post_id)
    post_img_repo = models.gift_imgs_repo.objects.filter(post__post_title=post.post_title)

    # Query all the images in post_img_repo
    post_img_list = []
    if post.post_img:
        post_img_list.append(str("http://fooxking.net:9000/media/" + str(post.post_img)))

    for imgs in post_img_repo:
        post_addr = str("http://fooxking.net:9000/media/" + str(imgs.more_img))
        post_img_list.append(post_addr)

    # Query all the related gifts
    related_gifts = []
    related_gifts_obj = post.related_gifts.all()
    for gift in related_gifts_obj:
        if gift.gift_img:
            gift_img = str("http://fooxking.net:9000/media/" + str(gift.gift_img)),
        else:
            gift_img = ''

        gift_dict = {
            'gift_id': gift.pk,
            'gift_name': gift.gift_name,
            'gift_image': gift_img,
            'gift_description': gift.gift_intro
        }
        related_gifts.append(gift_dict)

    post_dict = {
        'post_id': post.pk,
        'post_title': post.post_title,
        'post_image': post_img_list,
        'post_content': post.post_content,
        'related_gifts': related_gifts
    }
    # print(post_dict)
    return JsonResponse(post_dict)


def search_select(request):
    if request.body:
        request_content = json.loads(request.body)
        key_search = request_content.get("key_search")
        gift_price_start = request_content.get("gift_price_start")
        gift_price_end = request_content.get("gift_price_end")
        gift_gender = request_content.get("gift_gender")
        error_msg = ""
        found_gifts = models.Gift.objects.all()

        if key_search == "":
            error_msg = "No key_search"
        else:
            found_gifts = models.Gift.objects.filter(
                Q(gift_name__icontains=key_search) | Q(gift_intro__icontains=key_search))

        if gift_price_start == "":
            error_msg += " No gift_price_start"
        else:
            found_gifts = found_gifts.filter(tag_price__gte=gift_price_start)

        if gift_price_end == "":
            error_msg += " No gift_price_end"
        else:
            found_gifts = found_gifts.filter(tag_price__lte=gift_price_end)

        if gift_gender == "":
            error_msg += " No gift_gender"
        else:
            found_gifts = found_gifts.filter(tag_user=gift_gender)

        if found_gifts.count() == models.Gift.objects.count():
            msg = {
                'status': 2,
                'msg': "No found gifts",
                'gift_list': []
            }

            return JsonResponse(msg, status=422)
        else:
            found_gifts_json_list = []
            for gift in found_gifts:
                gift_dic = {
                    'gift_id': gift.pk,
                    'gift_name': gift.gift_name,
                    'gift_description': gift.gift_intro,
                    'gift_img': str("http://fooxking.net:9000/media/" + str(gift.gift_img)),
                    'tag_color': gift.tag_color,
                    'tag_festival': gift.tag_festival,
                    'tag_price': gift.tag_price,
                    'tag_user': gift.tag_user,
                    'tag_species': gift.tag_species,
                    'link': gift.link,
                    'num_thumb_up': gift.num_thumb_up,
                    "like_status": False,
                    "star_status": False
                }
                if request.user.is_authenticated:
                    if Like.objects.filter(person=request.user.person, gift_id=gift.pk).exists():
                        gift_dic["like_status"] = True
                    if Collect.objects.filter(person=request.user.person, gift_id=gift.pk).exists():
                        gift_dic["star_status"] = True
                found_gifts_json_list.append(gift_dic)
            msg = {
                'status': 1,
                'msg': error_msg,
                'gift_list': found_gifts_json_list
            }
            return JsonResponse(msg, status=200)
    else:
        msg = {
            'status': 2,
            'msg': "No requests",
            'gift_list': []
        }
        return JsonResponse(msg, status=422)


def recommend_personalized(request):
    if request.user.is_authenticated:
        def cosine_similarity(x, y):
            num = x.dot(y.T)
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            return num / denom

        # print(cosine_similarity(np.array([0,1,2,3,4]),np.array([1,2,1,4,5])))
        gift_list = []
        modified_list = []
        # Finding users similar to user A
        cosine_dict_user = {}
        person_obj = IdentityModels.Person.objects.all()
        for user in range(0, person_obj.count()):
            CosineSimilarity_user = []
            CosineSimilarity_user.clear()
            # 1 user_id
            user_pk = person_obj.get(id=user + 1).pk
            # print(user_pk)
            CosineSimilarity_user.append(user_pk)
            # 2 birthday
            birthday = person_obj.get(id=user_pk).birthday
            # print(birthday)
            if birthday:
                birthday = datetime.datetime.strptime(str(birthday), '%Y-%m-%d')
                birthday = time.mktime(birthday.timetuple())
                CosineSimilarity_user.append((birthday))
            else:
                birthday = 0
                CosineSimilarity_user.append((birthday))
            # 3 anniversary
            anniversary = person_obj.get(id=user_pk).anniversary
            if anniversary:
                anniversary = datetime.datetime.strptime(str(anniversary), '%Y-%m-%d')
                anniversary = time.mktime(anniversary.timetuple())
                CosineSimilarity_user.append(anniversary)
            else:
                anniversary = 0
                CosineSimilarity_user.append(anniversary)
            # 4 color
            favor_color = person_obj.get(id=user_pk).favor_color
            if favor_color:
                CosineSimilarity_user.append(len(favor_color))
                cosine_dict_user[user_pk] = CosineSimilarity_user
            else:
                CosineSimilarity_user.append(0)
            cosine_dict_user[user_pk] = CosineSimilarity_user
        # print(cosine_dict_user)
        # {user_id: [user_id, birthday, anniversary, color]}
        user_id = request.user.person.id
        # print(cosine_dict_user[user_id])
        X = np.array(cosine_dict_user[user_id])
        count = len(cosine_dict_user)
        # print(count)
        new_cosinesimilarity = {}
        for y in range(1, count + 1):
            # cosine_result = []
            Y = np.array(cosine_dict_user[y])
            similarity = float(cosine_similarity(X, Y))
            id_user = IdentityModels.Person.objects.get(pk=y).pk
            new_cosinesimilarity[id_user] = similarity
        # print(new_cosinesimilarity)
        # {similarity: id_user}
        value_descend = sorted(new_cosinesimilarity.items(), key=lambda e: e[1], reverse=True)
        # print(value_descend)
        top_ten_user = []
        for i in range(1, len(value_descend)):
            if len(value_descend) < 10:
                userID = value_descend[i][0]
                top_ten_user.append(userID)
            else:
                userID = value_descend[i][0]
                top_ten_user.append(userID)
                if i == 10:
                    break
        # print(top_ten_user)
        # the most similar users

        for i in range(0, len(top_ten_user) - 1):
            index = int(top_ten_user[i])
            like_obj = IdentityModels.Like.objects.filter(person_id=index).all()
            result = like_obj.filter(person_id=index)
            if result.exists():
                gift_id = like_obj.filter(person_id=index).first().gift_id
            else:
                continue
            gift_item = models.Gift.objects.get(pk=gift_id)
            gift_dict = {
                'gift_id': gift_item.pk,
                # 'gift_name': gift_item.gift_name,
                # 'gift_description': gift_item.gift_intro,
                # 'gift_img': str("http://fooxking.net:9000/media/" + str(gift_item.gift_img)),
                # 'tag_color': gift_item.tag_color,
                # 'tag_festival': gift_item.tag_festival,
                # 'tag_price': gift_item.tag_price,
                # 'tag_user': gift_item.tag_user,
                # 'tag_species': gift_item.tag_species,
                # 'link': gift_item.link,
                # 'num_thumb_up': gift_item.num_thumb_up,
            }
            gift_list.append(gift_dict)
        # print(gift_list)

        # Finding items similar to favorite items

        # 1. 创建用户A喜欢物品的向量
        like_obj = IdentityModels.Like.objects.filter(person_id=user_id).all()
        # print(like_obj)

        item_id_list = []
        for item in like_obj:
            item_id_list.append(item.gift.pk)
        # print(item_id_list)
        cosine_dict_item = {}
        for i in range(0, len(item_id_list)):
            CosineSimilarity = []
            CosineSimilarity.clear()
            index = item_id_list[i]
            result = like_obj.filter(gift_id=index)
            if result.exists():
                user_pk = like_obj.get(gift_id=index).person_id
            # print(user_pk)
            else:
                continue
            CosineSimilarity.append(index)
            item_id = like_obj.get(gift_id=index).gift_id
            price = models.Gift.objects.get(id=item_id).tag_price
            if price:
                price = float(price)
                price = int(price)
            # print(price)
            else:
                price = 0
            CosineSimilarity.append(price)

            species = models.Gift.objects.get(id=item_id).tag_species
            # print(species)
            if species:
                CosineSimilarity.append(len(species))
            else:
                CosineSimilarity.append(0)
            color = models.Gift.objects.get(id=item_id).tag_color
            if species:
                color = len(color)
                CosineSimilarity.append(color)
            else:
                CosineSimilarity.append(0)
            cosine_dict_item[item_id] = CosineSimilarity
        # print(cosine_dict_item)
        # {item_id: [item_id,price,species,color]}

        # 2. 创建Gift数据库中所有物品的向量
        gift_obj = models.Gift.objects.all()
        item_dict = {}
        # print(gift_obj)
        for gift in gift_obj:
            CosineSimilarity = []
            CosineSimilarity.clear()
            item_id = gift.pk
            CosineSimilarity.append(item_id)
            price = gift.tag_price
            if price:
                price = float(price)
                price = int(price)

            # print(price)
            else:
                price = 0
            CosineSimilarity.append(price)
            species = gift.tag_species
            # print(species)
            if species:
                CosineSimilarity.append(len(species))
            else:
                CosineSimilarity.append(0)
            color = gift.tag_color
            if color:
                CosineSimilarity.append(len(color))
            else:
                CosineSimilarity.append(0)
            item_dict[item_id] = CosineSimilarity
        # print(item_dict)
        # {item_id: [item_id, price, species, color]}

        # 3. 调用余弦相似性函数进行比较,排出前十个物品
        item_comparing = []
        for x in cosine_dict_item.items():
            index_comparing = x[1]
            X = np.array(index_comparing)
            item_comparing.append(X)
        # print(item_comparing)

        new_cosinesimilarity = {}
        for y in item_dict.items():
            for X in item_comparing:
                # cosine_result = []
                index_compared = y[1]
                Y = np.array(index_compared)
                similarity = float(cosine_similarity(X, Y))
                gift_id = y[0]
                new_cosinesimilarity[similarity] = gift_id
        # print(new_cosinesimilarity)
        # {similarity: id_user}
        value_descend = sorted(new_cosinesimilarity.items(), key=lambda e: e[0], reverse=True)
        # print(value_descend)
        top_ten_item = []
        for i in range(0, len(value_descend) - 1):
            item_id = value_descend[i][1]
            top_ten_item.append(item_id)
        # print(top_ten_item)
        for i in top_ten_item:  # 使用for in遍历出列表
            if not i in modified_list:  # 将遍历好的数字存储到控的列表中，因为使用了if not ，只有为空的的字符才会存里面，如果llist4里面已经有了，则不会存进去，这就起到了去除重复的效果！！
                modified_list.append(i)  # 把i存入新的列表中
        # print(modified_list)

        # 4. 将modified中数值对应的礼物信息返回给前端
        for i in range(0, len(modified_list)):
            index = int(modified_list[i])
            gift_id = models.Gift.objects.get(pk=index).pk
            gift_item = models.Gift.objects.get(pk=gift_id)
            gift_dict = {
                'gift_id': gift_item.pk,
                'gift_name': gift_item.gift_name,
                'gift_description': gift_item.gift_intro,
                'gift_img': str("http://fooxking.net:9000/media/" + str(gift_item.gift_img)),
                'tag_color': gift_item.tag_color,
                'tag_festival': gift_item.tag_festival,
                'tag_price': gift_item.tag_price,
                'tag_user': gift_item.tag_user,
                'tag_species': gift_item.tag_species,
                'link': gift_item.link,
                'num_thumb_up': gift_item.num_thumb_up,
            }
            gift_list.append(gift_dict)
        # print(gift_list)
        new_gift_list = []
        for i in gift_list:
            if not i in new_gift_list:
                new_gift_list.append(i)

        return JsonResponse(new_gift_list, safe=False)
    else:
        msg = {
            "status": 2,
            "msg": "account unauthorized"
        }
        return JsonResponse(msg, status=401)
