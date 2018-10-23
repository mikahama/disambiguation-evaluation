def parse_feature_to_dict(s):
	if s == "_":
		return {}
	return {_.split("=")[0] : _.split("=")[1] for _ in s.split("|")}

_partial_keys = ["Case", "Connegative", "VerbForm", "Mood", "Number", "Person", "Tense", "Voice"]