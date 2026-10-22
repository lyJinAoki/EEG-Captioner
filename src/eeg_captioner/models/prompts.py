QA_SYSTEM = (
    "You are an EEG signal analysis assistant. Answer the relational EEG "
    "multiple-choice question using only evidence supported by the supplied "
    "EEG segment. Return exactly the option letter and label, for example "
    "'A. delta', and no explanation."
)
JOINT_SYSTEM = (
    "You are an EEG signal analysis assistant. Follow the requested task using "
    "only evidence supported by the supplied EEG segment. For a multiple-choice "
    "perception question, return exactly the option letter and label with no "
    "explanation. For captioning, return exactly one <caption>...</caption> block."
)
CAPTION_QUESTION = (
    "Write a detailed physiological caption of the supplied EEG segment. "
    "Describe the dominant spectral activity, spatial distribution, and "
    "early-to-middle-to-late temporal changes supported by the signal. "
    "Return exactly one <caption>...</caption> block and no other text."
)
BEFORE_EEG = (
    "A 10-second, 20-channel EEG segment sampled at 200 Hz is represented "
    "by learned EEG embeddings below.\n<eeg>\n"
)


def prompt_parts(question, system_prompt=None):
    system_prompt = system_prompt or JOINT_SYSTEM
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{BEFORE_EEG}",
        f"\n</eeg>\n{question}<|im_end|>\n<|im_start|>assistant\n",
    )

