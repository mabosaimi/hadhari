from hadhari.preprocessing.preprocessor import (
    clean_text,
    normalize_arabic_letters,
    normalize_whitespace,
    preprocess_texts,
    reduce_repeated_characters,
    remove_diacritics,
    remove_punctuation,
    replace_emojis,
    replace_phone_numbers,
    replace_urls,
)


def test_normalize_arabic_letters():
    assert normalize_arabic_letters("أإآ") == "ااا"
    assert normalize_arabic_letters("يى") == "يي"
    assert normalize_arabic_letters("ة") == "ه"
    assert normalize_arabic_letters("ؤ") == "و"
    assert normalize_arabic_letters("ئ") == "ي"
    assert normalize_arabic_letters("تـطـويـل") == "تطويل"
    assert normalize_arabic_letters("") == ""


def test_remove_diacritics():
    text_with_diacritics = "مُحَمَّدٌ"
    assert remove_diacritics(text_with_diacritics) == "محمد"
    assert remove_diacritics("بدون") == "بدون"
    assert remove_diacritics("") == ""


def test_reduce_repeated_characters():
    assert reduce_repeated_characters("ممرررحبااا") == "مرحبا"
    assert reduce_repeated_characters("ووو") == "و"
    assert reduce_repeated_characters("عادي") == "عادي"
    assert reduce_repeated_characters("") == ""


def test_replace_urls():
    assert replace_urls("زرنا على http://example.com") == "زرنا على  <url> "
    assert replace_urls("موقع www.test.org") == "موقع  <url> "
    assert replace_urls("لا يوجد رابط هنا") == "لا يوجد رابط هنا"


def test_replace_phone_numbers():
    assert replace_phone_numbers("اتصل على 0123456789") == "اتصل على  <phone> "
    assert replace_phone_numbers("رقمي هو 966 50 1234567") == "رقمي هو  <phone> "
    assert replace_phone_numbers("ليس رقما 12345") == "ليس رقما 12345"  # Too short


def test_replace_emojis():
    assert replace_emojis("مرحبا 😀") == "مرحبا  <emoji> "
    assert replace_emojis("لا يوجد هنا") == "لا يوجد هنا"


def test_remove_punctuation():
    assert remove_punctuation("مرحبا، كيف حالك؟") == "مرحبا كيف حالك"
    assert remove_punctuation("رموز <url> و <phone>") == "رموز <url> و <phone>"  # Token tags should be kept
    assert remove_punctuation("!@#$%^&*()") == ""


def test_normalize_whitespace():
    assert normalize_whitespace("  مرحبا    بك  ") == "مرحبا بك"
    assert normalize_whitespace("\n\tسطر جديد  ") == "سطر جديد"
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""


def test_clean_text():
    # End-to-end testing
    complex_text = "   أهلاً بك يااااا مُحَمَّد، في موقعنا http://example.com 1234567890 😀!   "
    expected = "اهلا بك يا محمد في موقعنا <url> <phone> <emoji>"
    assert clean_text(complex_text) == expected

    # Edge cases
    assert clean_text("") == ""
    assert clean_text("    \n  ") == ""
    assert clean_text("نظيف") == "نظيف"


def test_preprocess_texts():
    texts = ["أهلاً", "مرحبا 😀", ""]
    result = preprocess_texts(texts)
    assert result == ["اهلا", "مرحبا <emoji>", ""]
