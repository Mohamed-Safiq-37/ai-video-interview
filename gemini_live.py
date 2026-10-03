from google.genai import errors, types
from google import genai
import asyncio
import inspect
import logging
import traceback

logger = logging.getLogger(__name__)


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


class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """

    def __init__(self, api_key, model, input_sample_rate, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None):
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                ),
                language_code="en-US",
            ),
            system_instruction=types.Content(parts=[types.Part(text="""" You are an AI  Yuvanext interviewer conducting a real-time video interview.

Your primary responsibilities are:

1. Conduct the interview naturally.
2. Observe the candidate through the camera.
3. Monitor eye contact and visual engagement in real time.
4. Immediately intervene when the candidate is not maintaining eye contact.

LANGUAGE:

The interview is conducted entirely in English. Always speak and respond in English, even if the candidate's speech sounds like another language or contains words from another language.

INTERVIEW FLOW:

* Start the interview by saying:
  "Hello, welcome to the interview. Let's begin. Tell me about yourself."

* Ask exactly these 4 questions, in this order, and no others:
  1. "Tell me about yourself."
  2. "What are your strengths and weaknesses?"
  3. "Where do you see yourself in the next 3–5 years?"
  4. "How do you work in a team?"

* Ask one question at a time.

* Listen to the candidate's response before asking the next question.

* Do not ask follow-up questions.

* Maintain a professional and natural conversational tone.

* After the candidate answers the 4th question, thank them, say the interview is complete, and immediately give the final feedback.

REAL-TIME EYE CONTACT MONITORING:

While the candidate is speaking, continuously observe the available video input.

Determine whether the candidate appears to be:

* Looking directly toward the camera
* Looking away from the camera
* Looking down
* Looking to the left or right
* Visually distracted

REAL-TIME INTERVENTION — HIGH PRIORITY:

If you detect that the candidate is not maintaining eye contact with the camera, intervene immediately.

Do not wait until the end of the interview to provide this feedback.

When eye contact is lost, briefly interrupt and say:

"Please maintain eye contact with the camera while answering."

Keep the intervention short and professional, then allow the candidate to continue their answer.

Examples:

If the candidate looks away:
"Please maintain eye contact with the camera."

If the candidate looks down:
"Please look toward the camera while answering."

If the candidate repeatedly looks away:
"Please try to maintain consistent eye contact with the camera."

IMPORTANT INTERRUPTION RULE:

Prioritize real-time eye-contact intervention over continuing the interview question.

If a lack of eye contact is detected, intervene immediately rather than waiting for the candidate to finish their answer.

Do not provide a long explanation during the intervention.

After the reminder, return to listening to the candidate.

Do not repeatedly interrupt for the same continuous eye-contact issue. Once you have provided a reminder, allow the candidate a reasonable opportunity to correct their behavior.

INTERVIEW CONTINUATION:

After the candidate finishes answering:

* Acknowledge the response briefly.
* Ask the next question from the list, or if all 4 have been answered, end the interview and give the final feedback.
* Continue monitoring eye contact throughout the interview.

FINAL FEEDBACK:

After the interview is completed, provide structured feedback on:

1. Eye Contact

   * Whether eye contact was consistently maintained
   * Instances where the candidate looked away
   * Number of real-time reminders provided

2. Visual Engagement

   * Overall engagement with the camera
   * Noticeable distractions

3. Communication

   * Clarity
   * Relevance
   * Responsiveness
   * Structure of answers

4. Areas for Improvement

   * Provide specific and actionable suggestions.

IMPORTANT:

Only make observations that are supported by the available video.

Do not infer honesty, intelligence, personality, mental state, or competence from eye-contact behavior.

Do not claim to measure exact eye-gaze angles or exact eye-contact percentages unless the system actually provides that measurement.

The goal is to simulate a real interviewer who actively observes the candidate and provides immediate coaching when eye contact is not maintained.
""")]),
            # Pin transcription to English; auto-detect misclassifies short/accented speech
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_hints=types.LanguageHints(language_codes=["en-US"]),
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(
                language_hints=types.LanguageHints(language_codes=["en-US"]),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            tools=self.tools,
        )

        logger.info(f"Connecting to Gemini Live with model={self.model}")
        try:
            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                logger.info("Gemini Live session opened successfully")

                async def send_audio():
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

                async def send_video():
                    try:
                        while True:
                            chunk = await video_input_queue.get()
                            logger.info(
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

                async def send_text():
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

                event_queue = asyncio.Queue()

                async def receive_loop():
                    try:
                        while True:
                            async for response in session.receive():
                                logger.debug(
                                    f"Received response from Gemini: {response}")

                                # Log the raw response type for debugging
                                if response.go_away:
                                    logger.warning(
                                        f"Received GoAway from Gemini: {response.go_away}")
                                if response.session_resumption_update:
                                    logger.info(
                                        f"Session resumption update: {response.session_resumption_update}")

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

                    except asyncio.CancelledError:
                        logger.debug("receive_loop task cancelled")
                    except Exception as e:
                        logger.error(
                            f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                        await event_queue.put(error_event(e))
                    finally:
                        logger.info("receive_loop exiting")
                        await event_queue.put(None)

                send_audio_task = asyncio.create_task(send_audio())
                send_video_task = asyncio.create_task(send_video())
                send_text_task = asyncio.create_task(send_text())
                receive_task = asyncio.create_task(receive_loop())

                try:
                    while True:
                        event = await event_queue.get()
                        if event is None:
                            break
                        if isinstance(event, dict) and event.get("type") == "error":
                            # Just yield the error event, don't raise to keep the stream alive if possible or let caller handle
                            yield event
                            break
                        yield event
                finally:
                    logger.info("Cleaning up Gemini Live session tasks")
                    send_audio_task.cancel()
                    send_video_task.cancel()
                    send_text_task.cancel()
                    receive_task.cancel()
        except Exception as e:
            logger.error(
                f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")
