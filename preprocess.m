function [documents] = preprocess(textdata)

% sanitise and tokenize

lowercase = lower(textdata);
documents = tokenizedDocument(lowercase);
documents = erasePunctuation(documents);
documents = removeStopWords(documents);
documents = regexprep(documents, '[^a-zA-Z]', '');

end