import inflect


class HOI(object):
    # TODO: no_interactions
    vowel_letters = ('a', 'u', 'e', 'i', 'o')
    no_interaction_string = 'no_interaction'

    def __init__(self, noun, verb, verb_ing, synonyms, definition, noun_is_plural):
        self.noun = noun.replace('_', ' ')
        self.noun_is_plural = noun_is_plural
        self.no_interaction = verb == self.no_interaction_string
        self.verb = verb.replace('_', ' ')
        self.verb_ing = self.__generate_ing_form(verb_ing)
        self.hoi_phrase = self.__generate_phrase()
        self.synonyms = synonyms
        self.definition = definition


    def __generate_ing_form(self, verb_ing):
        if self.no_interaction:
            return 'having no interaction with'
        else:
            return verb_ing.replace('_', ' ')

    def __generate_phrase(self):
        if self.noun_is_plural:
            return f"{self.verb_ing} {self.noun}"
        else:
            return f"{self.verb_ing} {inflect.engine().a(self.noun)}"
