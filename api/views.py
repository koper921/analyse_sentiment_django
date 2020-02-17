from django.shortcuts import render

from django.shortcuts import render, get_object_or_404, redirect
from rest_framework.views import APIView
from rest_framework.response import Response

import json
from .functions import classify_passenger, load_model, load_model_sent, classify_sentence, load_dict



class get_classification(APIView):
    def post(self, request):
        model = load_model_sent('./train_model/model.json', './train_model/model.h5')
        w2id = load_dict('./train_model/w2id.json')
        data = request.data
        prediction = classify_sentence(model, data, w2id, train=False)
        return (Response(prediction))
