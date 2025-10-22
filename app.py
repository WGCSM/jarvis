from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template, jsonify, session
from google import genai
from google.genai import types
import os

app = Flask(__name__)
# IMPORTANT: Session is needed to store the chat history. Set a secret key!
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_secret_key") 
client = genai.Client() # Assumes GEMINI_API_KEY is set in environment

# --- Function Calling Tools (J.A.R.V.I.S. Specialized Skills) ---


def check_health_status(metric: str) -> str:
    """Verifica el estado actual de una métrica de salud específica (ej: 'último movimiento', 'presión', 'glucosa')."""
    if "movimiento" in metric.lower():
        return "El último movimiento registrado fue hace 2 horas en la sala. Es hora de un recordatorio para caminar un poco."
    elif "presión" in metric.lower():
        return "Presión arterial registrada: 145/90. Ligeramente alta. Se recomienda descanso y reevaluación en 15 minutos."
    else:
        return f"Métrica de salud '{metric}' no monitoreada actualmente."

# Define las herramientas que Gemini puede usar
tools_list = [check_health_status]

def serialize_content(content: types.Content):
    """Converts a Content object into a safe dictionary for Flask session storage."""
    # Only store role and text part for simplicity
    if content and content.parts and content.parts[0].text:
        return {'role': content.role, 'text': content.parts[0].text}
    return None

def deserialize_content(data: dict):
    """Converts a dictionary back to a Content object for the Gemini API call."""
    if data and data['role'] and data['text']:
        return types.Content(role=data['role'], parts=[types.Part(text=data['text'])])
    return None

# --- Gemini Configuration ---

def get_gemini_response(user_input, chat_history):
    """Genera la respuesta de Gemini, incluyendo la lógica de Function Calling."""
    
    # Contexto clave para la Inclusión
    system_instruction = (
        "Eres J.A.R.V.I.S., un asistente virtual experto en cuidado, diseñado para personas mayores. "
        "Utiliza un lenguaje **cálido, simple y muy legible**. No uses abreviaturas ni jerga compleja. "
        "Tu principal objetivo es la seguridad y la comodidad del usuario. Usa las herramientas de Function Calling cuando sea apropiado."
    )
    
    # El historial incluye la instrucción del sistema
    contents = [
        # CORRECT: Use types.Part(text=...)
        *chat_history,
         types.Content(role="user", parts=[types.Part(text=user_input)])        
    ]
    
    try:
        # 1. Primera llamada a Gemini con la pregunta del usuario y las herramientas
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                    tools=tools_list,# FIX 2: Add system_instruction to the config
                    system_instruction=system_instruction
            ),
        )

        # 2. Manejo de Function Calling (si Gemini quiere usar una herramienta)
        if response.function_calls:
            function_call = response.function_calls[0]
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            # Ejecutar la función Python real
            if function_name == "check_health_status":
                function_result = check_health_status(**function_args)
            else:
                function_result = "Resultado: La función solicitada no está disponible."

            # 3. Segunda llamada a Gemini para obtener la respuesta conversacional
            contents.append(response.candidates[0].content) # Agregar la Function Call al historial
            contents.append(
                types.Content(role="function", parts=[
                    types.FunctionResponse(name=function_name, response={'result': function_result})
                ])
            )
            
            second_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools_list,# FIX 2: Add system_instruction to the config
                    system_instruction=system_instruction
                ),
            )
            return second_response.text

        # 4. Respuesta de texto simple (si no hubo Function Call)
        return response.text

    except Exception as e:
        return f"Lo siento, ocurrió un error en el sistema: {e}. Intenta de nuevo."

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    # Inicializa o limpia la sesión de chat al cargar la página
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('user_input')
    
    if not user_input:
        return jsonify({'error': 'No se recibió mensaje'}), 400

    # 1. FIX: Retrieve or initialize 'serialized_history' from the session
    # This line must be present and correctly defined before it is used later.
    serialized_history = session.get('chat_history', []) 

    # 2. Deserialize the history for Gemini (This uses serialized_history)
    gemini_history = [deserialize_content(item) for item in serialized_history]
    gemini_history = [item for item in gemini_history if item is not None]

    # Llama a Gemini
    ai_response_text = get_gemini_response(user_input, gemini_history)

    # 3. Create new content
    new_user_content = types.Content(role="user", parts=[types.Part(text=user_input)])
    new_model_content = types.Content(role="model", parts=[types.Part(text=ai_response_text)])

    # 4. Now we can safely use serialized_history.append()
    serialized_history.append(serialize_content(new_user_content))
    serialized_history.append(serialize_content(new_model_content))
    
    # Store the safe, serialized list back into the session
    session['chat_history'] = serialized_history
    
    return jsonify({'response': ai_response_text, 'user_input': user_input})

if __name__ == '__main__':
    app.run(debug=True)