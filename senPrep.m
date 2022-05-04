clear;
clc;

% load word embedding
emb = fastTextWordEmbedding;

% import lexicon table 
data = readLexicon;

% import lexicon and put into hashtable
% read positive word from lexicon
pos = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
% skip comments in the text file
C = textscan(pos,'%s','CommentStyle',';');
% convert cell array C to string
poswords = string(C{1});
% close files
fclose all;

% read negative word from lexicon
neg = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
% skip comments in the text file
C = textscan(neg,'%s','CommentStyle',';');
% convert cell array C to string
negwords = string(C{1});
% close files
fclose all;

% create hashtable
words_hash = java.util.Hashtable;

[possize, ~] = size(poswords); 
% Put all positive words in the hash-table
for ii = 1:possize
 words_hash.put(poswords(ii,1),1);
end

[negsize, ~] = size(negwords); 
% Put all negative words in the hash-table
for ii = 1:negsize
 words_hash.put(negwords(ii,1),-1);
end


% import test file
test = "train.csv";
testtable = readtable(test,'TextType','string');

% only use first x amount of rows
testtable = testtable(1:50000,:);

% prepare test text
text = testtable.SentimentText;
tetext = preprocess(text);

% remove words that arent in emb
idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

% remove words that arent in emb test
index = ~isVocabularyWord(emb,tetext.Vocabulary);
tetext = removeWords(tetext,index);

% convert words to word vectors
XTrain = word2vec(emb,data.Word);
YTrain = data.Label;

% convert words to word vectors test
XTest = word2vec(emb,tetext.Vocabulary);
YTest = testtable.Sentiment;

% train SVM to classify positive negative
model = fitcsvm(XTrain,YTrain);

% define vector for sentiment results
sentimentScore = zeros(size(tetext));
