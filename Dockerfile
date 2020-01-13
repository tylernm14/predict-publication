FROM ruby:2.6.5-buster

RUN mkdir /opt/predict-publication
WORKDIR /opt/predict-publication

RUN gem install bundler -v 2.1.4


RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py

RUN python3 -m pip install -U scikit-learn==0.22.1 pandas==0.25.3 nltk==3.3 joblib==0.14.1 \
    pillow==5.2.0 matplotlib==3.0.3 wordcloud==1.6.0

ADD Gemfile $HOME
ADD Gemfile.lock $HOME

RUN bundle install

ENV HOME /opt/predict-publication
ADD app.rb $HOME
ADD predict_publication.py $HOME
ADD predpub.pkl $HOME


ENV MODEL_MODULE=/opt/predict-publication/predict_publication.py

CMD ["bundle", "exec", "ruby", "app.rb", "-o", "0.0.0.0"]


