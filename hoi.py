class HOI(object):

    vowel_letters = ('a', 'u', 'e', 'i', 'o')

    def __init__(self, noun, verb, verb_ing, synonyms, definition):
        self.noun = noun
        self.verb = verb.replace('_', ' ')
        self.verb_ing = verb_ing.replace('_', ' ')
        self.article = self.__generate_article()
        self.synonyms = synonyms
        self.definition = definition

    def __generate_article(self):
        return 'an' if self.noun[0] in self.vowel_letters else 'a'
