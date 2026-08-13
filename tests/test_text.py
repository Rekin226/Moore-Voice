from moore_voice.text import (
    clean_text,
    detokenize,
    has_wiki_noise,
    is_fragment_pair,
    looks_moore,
    norm_text,
    pair_ok,
)


class TestDetokenize:
    def test_space_before_punct(self):
        assert detokenize("Bõe n zĩnd tʋm-tʋmdbã kũum poore ?") == \
            "Bõe n zĩnd tʋm-tʋmdbã kũum poore?"

    def test_spaced_hyphen_compound(self):
        assert detokenize("tʋm - tʋmdbã") == "tʋm-tʋmdbã"

    def test_french_apostrophe(self):
        assert detokenize("s ’ est produite .") == "s'est produite."

    def test_verse_reference(self):
        assert detokenize("Malaki 3 : 20 wilgame") == "Malaki 3: 20 wilgame"

    def test_open_paren(self):
        assert detokenize("yaa ( wa nag-bi )") == "yaa (wa nag-bi)"

    def test_idempotent_on_clean_text(self):
        s = "L'eau souterraine est essentielle, n'est-ce pas?"
        assert detokenize(s) == s


class TestLooksMoore:
    def test_diacritic_hit(self):
        assert looks_moore("Bõe n zĩnd tʋm-tʋmdbã kũum poore?")

    def test_function_words_no_diacritics(self):
        assert looks_moore("la woto yaa la ye")

    def test_english_rejected(self):
        assert not looks_moore("He will take them to Heaven.")

    def test_french_rejected(self):
        assert not looks_moore("Il est couché à plat ventre.")

    def test_empty(self):
        assert not looks_moore("")


class TestFilters:
    def test_fragment_pair(self):
        assert is_fragment_pair("Il est couché à plat ventre.", "yãoogo")
        assert not is_fragment_pair("Bonjour à tous.", "Ne y windga.")

    def test_wiki_noise(self):
        assert has_wiki_noise("Seb-kãngã paama leeb naoor a ye {{PLURAL:$1}}")
        assert has_wiki_noise("Lebe tʋgle ni $1")
        assert not has_wiki_noise("Seb-kãngã paama leeb naoor a ye")

    def test_pair_ok_rejects_numeric_only(self):
        assert not pair_ok("3 : 18 .", "3 : 18 , MN .")

    def test_pair_ok_rejects_copy(self):
        assert not pair_ok("Same text", "same text")

    def test_pair_ok_accepts_normal(self):
        assert pair_ok("The children are going to school.", "Kambã na n kẽnga lekollẽ.")

    def test_pair_ok_rejects_too_long(self):
        assert not pair_ok("word " * 201, "Kambã na n kẽnga lekollẽ.")


class TestNormText:
    def test_nbsp_and_zero_width(self):
        assert norm_text("a b​c") == "a bc"

    def test_clean_strips_urls(self):
        assert clean_text("see https://x.com/page now") == "see now"
