from google.genai import errors, types
from google import genai
from collections import Counter
import asyncio
import inspect
import logging
import traceback

logger = logging.getLogger(__name__)

# Interview languages selectable from the UI. Questions are fixed per language so
# every candidate is asked exactly the same thing.
LANGUAGES = {
    "en": {
        "name": "English",
        "code": "en-US",
        "greeting": "Hello, welcome to the interview. Let's begin. Tell me about yourself.",
        "questions": [
            "Tell me about yourself.",
            "What are your strengths and weaknesses?",
            "Where do you see yourself in the next 3–5 years?",
            "How do you work in a team?",
        ],
    },
    "ta": {
        "name": "Tamil",
        "code": "ta-IN",
        "greeting": "வணக்கம், நேர்காணலுக்கு வரவேற்கிறோம். தொடங்கலாம். உங்களைப் பற்றி சொல்லுங்கள்.",
        "questions": [
            "உங்களைப் பற்றி சொல்லுங்கள்.",
            "உங்கள் பலங்கள் மற்றும் பலவீனங்கள் என்ன?",
            "அடுத்த 3 முதல் 5 ஆண்டுகளில் உங்களை எங்கே பார்க்கிறீர்கள்?",
            "ஒரு குழுவில் நீங்கள் எப்படி வேலை செய்வீர்கள்?",
        ],
    },
}
DEFAULT_LANGUAGE = "en"

# USD per 1M tokens, paid tier (https://ai.google.dev/gemini-api/docs/pricing, Gemini 3.8 Live).
# Only used for the cost estimate. Thinking tokens are billed at the audio output rate.
PRICE_PER_M_INPUT = {"AUDIO": 3.00, "IMAGE": 1.00, "VIDEO": 1.00, "TEXT": 0.75}
PRICE_PER_M_OUTPUT = {"AUDIO": 12.00, "TEXT": 4.50, "THOUGHTS": 12.00}

# Reconnect attempts allowed in a row before giving up on a dropped connection
MAX_RESUME_ATTEMPTS = 3

# Testing only: cycle through the questions again and again until the candidate
# says "finish interview", instead of stopping after the last question.
REPEAT_QUESTIONS_UNTIL_FINISH = True


def build_system_instruction(language):
    lang = LANGUAGES[language]
    name = lang["name"]
    questions = "\n".join(f'  {i}. "{q}"' for i, q in enumerate(lang["questions"], 1))

    if REPEAT_QUESTIONS_UNTIL_FINISH:
        question_rule = f"""* Ask these 4 questions, in this order, and no others:
{questions}
* After the 4th question, start again from question 1 and keep repeating the same 4 questions in the same order. Do not end the interview on your own."""
        end_rule = """* The interview ends only when the candidate says "finish interview" (or the same meaning in any language). Then stop asking questions, thank them, say the interview is complete, and immediately give the final feedback, covering all of their answers."""
    else:
        question_rule = f"""* Ask exactly these 4 questions, in this order, and no others:
{questions}"""
        end_rule = "* After the candidate answers the 4th question, thank them, say the interview is complete, and immediately give the final feedback."

    return f"""You are an AI Yuvanext interviewer conducting a real-time video interview.

Your responsibilities are:

1. Conduct the interview naturally.
2. Silently observe the candidate's answers, voice and video throughout the interview.
3. Give all feedback only once, at the end of the interview.

LANGUAGE:

The interview language is {name}. Always speak and respond only in {name} — questions, acknowledgements and the final feedback.
If the candidate speaks in another language, or mixes languages, still continue in {name}. Do not switch languages and do not ask the candidate to switch.

INTERVIEW FLOW:

* Start the interview by saying:
  "{lang["greeting"]}"

{question_rule}

* Ask one question at a time.
* Listen to the candidate's full response before asking the next question.
* Candidates may pause to think in the middle of an answer. A short pause does not mean the answer is over.
* Do not ask follow-up questions.
* Maintain a professional and natural conversational tone.
* Between questions, acknowledge the response with a brief, neutral phrase only. Do not evaluate or praise the answer.
{end_rule}

NO INTERRUPTIONS DURING THE INTERVIEW:

Never interrupt the candidate and never comment on their behaviour while the interview is in progress.
Do not remind them about eye contact, posture, pauses or anything else during the interview.
Silently note your observations and save them for the final feedback.

WHAT TO OBSERVE:

A. Answer content
  1. Relevance — does the answer address the question that was asked?
  2. Completeness and depth — is the answer sufficiently developed?
  3. Structure — is there a clear, logical flow (for experiences: situation, action, result)?
  4. Specific examples — are claims backed by concrete examples?
  5. Conciseness — does the candidate stay on point without rambling or repeating?

B. Verbal delivery
  6. Clarity and articulation — is the speech easy to understand?
  7. Fluency — long pauses in the middle of answers, hesitation, filler words (such as "um", "uh", "like").
  8. Response time — how quickly the candidate starts answering after a question.
  9. Pace and volume — too fast, too slow, too quiet?
  10. Confidence and tone — steady, assured voice versus uncertain or monotone.
  11. Language use — grammar, vocabulary, and whether the candidate stayed in {name}.

C. Non-verbal (from the video)
  12. Eye contact — looking toward the camera versus looking away, down or to the side.
  13. Facial expression and engagement — attentive, friendly expression.
  14. Posture and body language — upright posture, fidgeting, distracting gestures.
  15. Distractions and setup — looking at other screens or a phone, background noise, lighting, framing.

FINAL FEEDBACK:

After the interview is complete, give structured feedback in {name}:

1. For each group (Answer content, Verbal delivery, Non-verbal), give a rating out of 5 for each factor with one short sentence explaining it, referring to specific moments or questions where possible.
2. An overall score out of 10.
3. The candidate's top 3 strengths.
4. The top 3 specific, actionable improvements.

Keep each point short — the feedback is spoken aloud.

IMPORTANT:

Only make observations that are supported by what you actually heard and saw.
If the camera was off or the video was unclear, say that the non-verbal factors could not be assessed instead of guessing.
Do not infer honesty, intelligence, personality, mental state or competence from non-verbal behaviour.
Do not claim exact measurements (such as eye-contact percentages or pause durations in seconds).
"""


def error_event(e):
    """Builds a client-facing error event from an exception raised by the Gemini API."""
    code = None
    if isinstance(e, errors.APIError):
        code = e.code
        message = e.message or (e.details if isinstance(e.details, str) else str(e))
    else:
        message = str(e) or type(e).__name__

    lowered = message.lower()
    if code == 429 or "quota" in lowered or "resource_exhausted" in lowered or "rate limit" in lowered:
        title = "Gemini API quota exhausted"
    elif code in (401, 403) or "api key" in lowered or "permission" in lowered:
        title = "Gemini API authentication failed"
    else:
        title = "Gemini API error"

    return {"type": "error", "title": title, "error": message, "code": code}


def _modality_name(modality):
    return modality.name if hasattr(modality, "name") else str(modality)


class UsageTracker:
    """Accumulates token usage reported by the Live API and estimates the session cost."""

    def __init__(self):
        self.messages = 0
        self.input_tokens = Counter()
        self.output_tokens = Counter()

    def add(self, usage):
        self.messages += 1
        prompt_tokens = Counter()
        for detail in usage.prompt_tokens_details or []:
            prompt_tokens[_modality_name(detail.modality)] += detail.token_count or 0
        # Tokens the API didn't break down by modality are counted as audio (the priciest input)
        unaccounted = (usage.prompt_token_count or 0) - sum(prompt_tokens.values())
        if unaccounted > 0:
            prompt_tokens["AUDIO"] += unaccounted

        response_tokens = Counter()
        for detail in usage.response_tokens_details or []:
            response_tokens[_modality_name(detail.modality)] += detail.token_count or 0
        unaccounted = (usage.response_token_count or 0) - sum(response_tokens.values())
        if unaccounted > 0:
            response_tokens["AUDIO"] += unaccounted
        # Thinking tokens are only billed separately when they aren't already in response_token_count
        billed_total = (usage.prompt_token_count or 0) + (usage.response_token_count or 0)
        if usage.thoughts_token_count and (usage.total_token_count or 0) > billed_total:
            response_tokens["THOUGHTS"] += usage.thoughts_token_count

        self.input_tokens.update(prompt_tokens)
        self.output_tokens.update(response_tokens)

        logger.info(
            f"Usage: input={dict(prompt_tokens)} output={dict(response_tokens)} "
            f"total={usage.total_token_count} | session cost so far ≈ ${self.cost():.4f}")

    def cost(self):
        total = sum(n * PRICE_PER_M_INPUT.get(m, PRICE_PER_M_INPUT["AUDIO"])
                    for m, n in self.input_tokens.items())
        total += sum(n * PRICE_PER_M_OUTPUT.get(m, PRICE_PER_M_OUTPUT["AUDIO"])
                     for m, n in self.output_tokens.items())
        return total / 1_000_000

    def as_event(self):
        return {
            "type": "usage",
            "input_tokens": dict(self.input_tokens),
            "output_tokens": dict(self.output_tokens),
            "cost_usd": round(self.cost(), 6),
        }

    def log_summary(self):
        if not self.messages:
            return
        logger.info(
            f"Session usage summary: {self.messages} usage reports | "
            f"input={dict(self.input_tokens)} output={dict(self.output_tokens)} | "
            f"estimated cost ≈ ${self.cost():.4f}")


class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """

    def __init__(self, api_key, model, input_sample_rate, language=DEFAULT_LANGUAGE, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            language (str, optional): Interview language key from LANGUAGES. Defaults to English.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.language = language if language in LANGUAGES else DEFAULT_LANGUAGE
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    def _build_config(self, resume_handle=None):
        language_code = LANGUAGES[self.language]["code"]
        # Pin transcription to the interview language; auto-detect misclassifies short/accented speech.
        # Tamil speakers often mix in English, so English is kept as a secondary hint.
        input_hints = [language_code] if language_code == "en-US" else [language_code, "en-US"]

        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                ),
                language_code=language_code,
            ),
            system_instruction=types.Content(parts=[types.Part(text=build_system_instruction(self.language))]),
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_hints=types.LanguageHints(language_codes=input_hints),
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(
                language_hints=types.LanguageHints(language_codes=[language_code]),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
                # Give candidates time to pause and think without the turn being cut off
                automatic_activity_detection=types.AutomaticActivityDetection(
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    silence_duration_ms=1500,
                ),
            ),
            # Safety net only: the end-of-interview feedback needs the whole interview in context,
            # so compression kicks in just before the 128k window would end the session.
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=120_000,
                sliding_window=types.SlidingWindow(target_tokens=90_000),
            ),
            # The server resets the connection roughly every 10 minutes; resumption carries the
            # session (and its context) over to a new connection.
            session_resumption=types.SessionResumptionConfig(handle=resume_handle),
            tools=self.tools,
        )

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None):
        event_queue = asyncio.Queue()
        usage = UsageTracker()
        resume_handle = None

        async def send_audio(session):
            try:
                while True:
                    chunk = await audio_input_queue.get()
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                    )
            except asyncio.CancelledError:
                logger.debug("send_audio task cancelled")
            except Exception as e:
                logger.error(
                    f"send_audio error: {e}\n{traceback.format_exc()}")

        async def send_video(session):
            try:
                while True:
                    chunk = await video_input_queue.get()
                    logger.debug(
                        f"Sending video frame to Gemini: {len(chunk)} bytes")
                    await session.send_realtime_input(
                        video=types.Blob(
                            data=chunk, mime_type="image/jpeg")
                    )
            except asyncio.CancelledError:
                logger.debug("send_video task cancelled")
            except Exception as e:
                logger.error(
                    f"send_video error: {e}\n{traceback.format_exc()}")

        async def send_text(session):
            try:
                while True:
                    text = await text_input_queue.get()
                    logger.info(f"Sending text to Gemini: {text}")
                    await session.send_realtime_input(text=text)
            except asyncio.CancelledError:
                logger.debug("send_text task cancelled")
            except Exception as e:
                logger.error(
                    f"send_text error: {e}\n{traceback.format_exc()}")

        async def receive_loop(session):
            """Processes server messages until the connection should be resumed.

            Returns True when a GoAway was received and the current turn has finished,
            so the session can move to a new connection without cutting the model off.
            """
            nonlocal resume_handle
            go_away_received = False

            while True:
                async for response in session.receive():
                    logger.debug(
                        f"Received response from Gemini: {response}")

                    if response.go_away:
                        logger.warning(
                            f"Received GoAway from Gemini: {response.go_away}")
                        go_away_received = True

                    update = response.session_resumption_update
                    if update and update.resumable and update.new_handle:
                        resume_handle = update.new_handle

                    if response.usage_metadata:
                        usage.add(response.usage_metadata)
                        # Running totals, so the UI has the latest figure whenever the session ends
                        await event_queue.put(usage.as_event())

                    server_content = response.server_content
                    tool_call = response.tool_call

                    if server_content:
                        if server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                if part.inline_data:
                                    if inspect.iscoroutinefunction(audio_output_callback):
                                        await audio_output_callback(part.inline_data.data)
                                    else:
                                        audio_output_callback(
                                            part.inline_data.data)

                        if server_content.input_transcription and server_content.input_transcription.text:
                            await event_queue.put({"type": "user", "text": server_content.input_transcription.text})

                        if server_content.output_transcription and server_content.output_transcription.text:
                            await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})

                        if server_content.interaction_status:
                            status_val = server_content.interaction_status
                            status_str = status_val.name if hasattr(
                                status_val, 'name') else str(status_val)
                            logger.debug(
                                f"Interaction status: {status_str}")
                            await event_queue.put({"type": "interaction_status", "status": status_str})

                        if server_content.turn_complete:
                            await event_queue.put({"type": "turn_complete"})

                        if server_content.interrupted:
                            if audio_interrupt_callback:
                                if inspect.iscoroutinefunction(audio_interrupt_callback):
                                    await audio_interrupt_callback()
                                else:
                                    audio_interrupt_callback()
                            await event_queue.put({"type": "interrupted"})

                    if tool_call:
                        function_responses = []
                        for fc in tool_call.function_calls:
                            func_name = fc.name
                            args = fc.args or {}

                            if func_name in self.tool_mapping:
                                try:
                                    tool_func = self.tool_mapping[func_name]
                                    if inspect.iscoroutinefunction(tool_func):
                                        result = await tool_func(**args)
                                    else:
                                        loop = asyncio.get_running_loop()
                                        result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                except Exception as e:
                                    result = f"Error: {e}"

                                function_responses.append(types.FunctionResponse(
                                    name=func_name,
                                    id=fc.id,
                                    response={"result": result}
                                ))
                                await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})

                        await session.send_tool_response(function_responses=function_responses)

                # session.receive() iterator ended (e.g. after turn_complete) — re-enter to keep listening
                logger.debug(
                    "Gemini receive iterator completed, re-entering receive loop")
                if go_away_received and resume_handle:
                    return True

        async def run_connections():
            """Keeps the session alive across server-side connection resets."""
            failed_attempts = 0
            try:
                while True:
                    is_resume = resume_handle is not None
                    logger.info(
                        f"{'Resuming' if is_resume else 'Connecting to'} Gemini Live with "
                        f"model={self.model}, language={self.language}")
                    try:
                        async with self.client.aio.live.connect(model=self.model, config=self._build_config(resume_handle)) as session:
                            logger.info("Gemini Live session opened successfully")
                            failed_attempts = 0
                            senders = [
                                asyncio.create_task(send_audio(session)),
                                asyncio.create_task(send_video(session)),
                                asyncio.create_task(send_text(session)),
                            ]
                            try:
                                await receive_loop(session)
                            finally:
                                for task in senders:
                                    task.cancel()
                        logger.info("Moving Gemini Live session to a new connection")
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # A dropped connection can be resumed if the server gave us a handle
                        if resume_handle and failed_attempts < MAX_RESUME_ATTEMPTS:
                            failed_attempts += 1
                            logger.warning(
                                f"Gemini connection lost ({type(e).__name__}: {e}); "
                                f"resuming (attempt {failed_attempts}/{MAX_RESUME_ATTEMPTS})")
                            await asyncio.sleep(1)
                            continue
                        logger.error(
                            f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                        await event_queue.put(error_event(e))
                        return
            except asyncio.CancelledError:
                logger.debug("run_connections task cancelled")
            finally:
                logger.info("Gemini connection loop exiting")
                await event_queue.put(None)

        connection_task = asyncio.create_task(run_connections())
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
                if isinstance(event, dict) and event.get("type") == "error":
                    break
        finally:
            logger.info("Cleaning up Gemini Live session tasks")
            connection_task.cancel()
            usage.log_summary()
            logger.info("Gemini Live session closed")
