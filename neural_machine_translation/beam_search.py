# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import string
import random
from data_utils import *
from rnn import *
import torch
import codecs
from tqdm import tqdm
import string
import math

#Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load vocabulary files
input_lang = torch.load('data-bin/fra.data')
output_lang = torch.load('data-bin/eng.data')

#Create and empty RNN model
encoder = EncoderRNN(input_size=input_lang.n_words, device=device)
attn_decoder = AttnDecoderRNN(output_size=output_lang.n_words, device=device)

#Load the saved model weights into the RNN model
encoder.load_state_dict(torch.load('model/encoder'))
attn_decoder.load_state_dict(torch.load('model/decoder'))

#Return the decoder output given input sentence 
#Additionally, the previous predicted word and previous decoder state can also be given as input
def translate_single_word(encoder, decoder, sentence, decoder_input=None, decoder_hidden=None, max_length=MAX_LENGTH, device=device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        if decoder_input==None:
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        else:
            decoder_input = torch.tensor([[output_lang.word2index[decoder_input]]], device=device) 
        
        if decoder_hidden == None:        
            decoder_hidden = encoder_hidden
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        return decoder_output.data, decoder_hidden

#########################################################################################
#####Modify this function to use beam search to predict instead of greedy prediction#####
#########################################################################################
def beam_search(encoder,decoder,input_sentence,beam_size=1,max_length=MAX_LENGTH):
    decoded_output = [[] for x in range(beam_size)]
    decoded_output_heuristics = [0 for x in range(beam_size)]
    previous_decoded_outputs = []
    #Predicted the first word
    decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, decoder_input=None, decoder_hidden=None)
    
    #Get the probability of all output words
    decoder_output_probs = decoder_output.data
    
    #Select the id of the word with maximum probability
    #idx = torch.argmax(decoder_output_probs)

    #Select the ids of words with top k probabilities
    probs = torch.topk(decoder_output_probs, beam_size)[0][0]
    idxs = torch.topk(decoder_output_probs, beam_size)[1][0]
    
    #Convert the predicted id to the word
    #first_word = output_lang.index2word[idx.item()]
    
    #Convert the predicted ids to the words
    words = []
    for i in idxs.tolist():
      words.append(output_lang.index2word[i])

    #Add the predicted word to the output list and also set it as the previous prediction
    #decoded_output.append(first_word)
    #previous_decoded_output = first_word

    #Add the predicted word to the output list and also set it as the previous prediction
    for i in range(beam_size):
      decoded_output[i].append(words[i])
      decoded_output_heuristics[i] += probs.tolist()[i]
    previous_decoded_outputs = words
    
    #Loop until the maximum length
    for i in range(max_length):
    
      #Predict the next word given the previous prediction and the previous decoder hidden state
      #decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, previous_decoded_output, decoder_hidden)

      shorten_index = False
      for j in range(beam_size):
        if shorten_index:
          j -= 1
          shorten_index = False
        
        decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, previous_decoded_outputs[j], decoder_hidden)

        #Get the probability of all output words
        decoder_output_probs = decoder_output.data

        #Select the ids of words with top k probabilities
        probs = torch.topk(decoder_output_probs, beam_size)[0][0]
        idxs = torch.topk(decoder_output_probs, beam_size)[1][0]

        #Convert the predicted ids to the words
        words = []
        for k in idxs.tolist():
          if k == EOS_token:
            shorten_index = True
            continue
          else:
            words.append(output_lang.index2word[k])

        for k in range(len(words)):
          decoded_output[k].append(words[k])
          decoded_output_heuristics[k] += probs.tolist()[k]
        previous_decoded_outputs = words
            
        #Get the probability of all output words
        #decoder_output_probs = decoder_output.data
        
        #Select the id of the word with maximum probability
        #idx = torch.argmax(decoder_output_probs)
        
        #Break if end of sentence is predicted
        #if idx.item() == EOS_token:
            #break 
            
        #Else add the predicted word to the list
        #else:
            #Convert the predicted id to the word
            #selected_word = output_lang.index2word[idx.item()]
            
            #Add the predicted word to the output list and update the previous prediction
            #decoded_output.append(selected_word)    
            #previous_decoded_output = selected_word
            
    #Convert list of predicted words to a sentence and detokenize 
    max_index = decoded_output_heuristics.index(max(decoded_output_heuristics))
    output_translation = ""
    for i in range(len(decoded_output[max_index])):
      if i > 0 and (decoded_output[max_index][i] == "." or decoded_output[max_index][i] == "?") and (decoded_output[max_index][i-1] == "." or decoded_output[max_index][i-1] == "?"):
        break
      else:
        if i == 0:
          output_translation += decoded_output[max_index][i]
        else:
          output_translation += " " + decoded_output[max_index][i]
    return output_translation


with open("data/test.eng", "r") as f:
  target_sentences = f.readlines()

with open("data/test.fra", "r") as f:
  source_sentences = f.readlines()

target = codecs.open('example.txt','w',encoding='utf-8')

beam_size = 2
for i,source_sentence in enumerate(source_sentences):

    target_sentence = normalizeString(target_sentences[i])
    input_sentence = normalizeString(source_sentence)
    
    hypothesis = beam_search(encoder, attn_decoder, input_sentence, beam_size=beam_size)
    
    print("S-"+str(i)+": "+input_sentence)
    print("T-"+str(i)+": "+target_sentence)
    print("H-"+str(i)+": "+hypothesis)
    print()
    target.write(hypothesis+'\n')
target.close()    
