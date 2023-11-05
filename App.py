import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

async def obtener_valor():
    response = await asyncio.to_thread(requests.get, "http://localhost:3000/obtener-valor")
    
    if response.status_code == 200:
        data = response.json()
        valor = data.get("valorSet")
        print("Valor obtenido desde el servidor:", valor)
        return valor
    else:
        print("Error al obtener el valor desde el servidor.")
        return None

async def main():
    valor = await obtener_valor()
    
    if valor is not None:
        tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1_5-3b', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1_5-3b', trust_remote_code=True)

        x = tokenizer.encode(valor, return_tensors='pt')
        y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

        # decoding
        generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("Código generado:", generated_code)

if __name__ == "__main__":
    asyncio.run(main())
