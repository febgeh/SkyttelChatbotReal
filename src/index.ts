import OpenAI from "openai";
import 'dotenv/config';
import readline from 'readline';
import similarity from "compute-cosine-similarity";

//Importerer Q&A datasettet fra questions.ts
import { dataset } from "./questions.js"; 


// Bruker typescript så lager interface for chatbotten
interface chat {
    role: "system" | "user" | "assistant",
    content: string
}

//henter openai api key fra .env filen
const openai = new OpenAI({
    apiKey: process.env["OPENAI_API_KEY"] || "No key found",
});

//lager et interface for å kunne kommunisere med brukeren
const userInterface = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

//lager en funksjon som lar brukeren skrive til chatbotten
const askUser = async () => {
    return new Promise((resolve) => {
        userInterface.question('Spør Skyttel Chatbot: ', (input) => {
            resolve(input);
        });
    });
};

//lager en funksjon som lar chatbotten skrive til brukeren. 
async function chatbot (input: string, chatHistory: chat[]): Promise<{response: string, chatHistory: chat[]}> {
    userInterface.prompt();

    //lager en chatbot respons ved hjelp av openai sitt chat api og bruker chatHistory for å holde styr på samtalen
    const res = await openai.chat.completions.create({
        model: "gpt-3.5-turbo", 
        messages: [
            ...chatHistory,
            {
                role: "system",
                content: "System wiki: " + await similarWiki(input),
            },
            {
                role: "user",
                content: input,
            },
        ]
    })
    //returnerer chatbotten sin respons og chatHistory
    return {
        response: res.choices[0].message.content,
        chatHistory: [
            ...chatHistory,
            {
                role: "system",
                content: "System wiki: " + await similarWiki(input),
            },
            {
                role: "user",
                content: input,
            },
            {
                role: "assistant",
                content: res.choices[0].message.content,
            },
        ],
    }
}

//lager en funksjon som finner det mest lignende paragrafet til inputen
async function similarWiki(input: string) {
    const inputUserEmbed = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: input,
    })

    //lager en liste med lignendehts score for alle paragrafene
    const questionVectorEmbedding = inputUserEmbed.data[0].embedding;
    let wiki_Embeds = await Promise.all(dataset.map(async (wiki) => {
        const QandA_Embed = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: wiki,
        })
        const questionEmbedding = QandA_Embed.data[0].embedding;
        return {embed: questionEmbedding, wiki }
    }))

    //returner scoren
    const similarityScores = wiki_Embeds.map((wiki) => {
        return {
            similarity: similarity(questionVectorEmbedding, wiki.embed),
            wiki: wiki.wiki
        }
    })

    //sorterer listen med lignendehts score og returnerer paragrafen med høyest score
    similarityScores.sort((a, b) => b.similarity - a.similarity);

    return similarityScores[0].wiki
}

//Gir chatbotten en oppgave
let chat: chat[]= [
    {
        role: "system",
        content: "Oppgaven din er å være en chatbot for bedriften Skyttel AS. Du er en hjelpsom assistent som svarer på sprøsmål med bruk av infromasjonene fra wikien nedefor. Hvis du ikke har svaret på et spørsmål ber du brukeren utdype spørsmålet eller si du ikke har svaret og referer brukeren til kundesenteret."
    }
]

//lager en main funksjon som lar brukeren skrive til chatbotten og chatbotten skrive til brukeren
async function main (){
    while(true) {
        const input = await askUser() as string
        const newInput = input.toLowerCase();
    
        //sjekker om brukeren vil avslutte programmet
        switch(newInput) {
            case "stop":
            case "exit":
            case "quit":
            case "hade":
                userInterface.close();
                console.log("Goodbye!");
                console.log(chat);
                return;
            default:
                
                //hvis brukeren ikke vil avslutte programmet lar vi chatbotten svare
                const { response, chatHistory } = await chatbot(input, chat);
                //printer chatbotten sin respons
                console.log(response)
                chat = chatHistory
                break;
        }
    }
}

main()