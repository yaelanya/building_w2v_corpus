import argparse

import yaml
from gensim.models.word2vec import PathLineSentences, Word2Vec


def main(args):
    # load a config file
    with open(args.param_path) as f:
        param = yaml.safe_load(f)['w2v']

    sentences = PathLineSentences(param['input'])

    model = Word2Vec(
            sentences, 
            size=param['size'], 
            window=param['window'], 
            min_count=param['min_count'],
            sg=param['sg'],
            negative=param['negative'],
            iter=param['iter'],
            workers=param['workers']
        )
    
    model.save(param['output'])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('param_path', type=str)
    args = parser.parse_args()

    main(args)
