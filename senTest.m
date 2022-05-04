clc;

for ii = 1 : tetext.length
    docwords = tetext(ii).Vocabulary;
    for jj = 1 : length(docwords)
        if words_hash.containsKey(docwords(jj))
            sentimentScore(ii) = sentimentScore(ii) +  words_hash.get(docwords(jj));
        end
    end
    if sentimentScore(ii) == 0
        vec = word2vec(emb,docwords);
        [~,scores] = predict(model,vec);
        sentimentScore(ii) = mean(scores(:,1));
        if isnan(sentimentScore(ii))
            sentimentScore(ii) = 0;
        end
    end
    if sentimentScore(ii) ~= 0
        fprintf('+++Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(tetext(ii)), sentimentScore(ii), YTest(ii));
    else
        fprintf('---Sent: %d, words: %s, Not Covered, GoldScore: %d\n', ii, joinWords(tetext(ii)),  YTest(ii));
    end
end

% coverage and accuracy
% Find number of all sentiment score = 0 :not found
zeroval = sum(sentimentScore == 0);
% Find all distinct values
covered = numel(sentimentScore) - zeroval;
% calculate true positives and true negatives (coverage)
fprintf("Total of positive and negative classes (coverage): %2.2f%%, Distinct %d, Not Found or Neutral: %d\n", (covered * 100)/numel(sentimentScore), covered, zeroval);

% calculate true positives and true negatives
tp = sentimentScore((sentimentScore > 0) & ( YTest > 0));
tn = sentimentScore((sentimentScore  < 0) &( YTest == 0));
fp = sentimentScore((sentimentScore > 0) & ( YTest == 0));
fn = sentimentScore((sentimentScore  < 0) &( YTest > 0));

% calculate accuracy
acc = (numel(tp) + numel(tn))/covered;
fprintf("Accuracy: %2.2f%%, true positive: %d, true negative: %d\n", acc*100, numel(tp), numel(tn));

% calculate precision
prec = numel(tp)/(numel(tp) + numel(fp));
fprintf("Precision: %2.2f%%, true positive: %d, false positive: %d\n", prec*100, numel(tp), numel(fp));

% calculate recall
recall = numel(tp)/(numel(tp) + numel(fn));
fprintf("Recall: %2.2f%%, true positive: %d, false negative: %d\n", recall*100, numel(tp), numel(fn));

% calculate F1-measure
f1m = (2 * prec * recall)/(prec + recall);
fprintf("F1-measure: %2.2f%%", f1m*100);

% predict sentiment
[YPred,scores] = predict(model,XTest); 

% visualise in word cloud
words = tetext.Vocabulary;

figure
subplot(1,2,1)
idx = YPred == "1";
wordcloud(words(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")
