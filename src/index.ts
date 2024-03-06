import OpenAI from "openai";
import 'dotenv/config';
import readline from 'readline';
import similarity from "compute-cosine-similarity";

//Q&A
import { dataset } from "./questions.js"; 


// OpenAI chatbot
interface chat {
    role: "system" | "user" | "assistant",
    content: string
}

const openai = new OpenAI({
    apiKey: process.env["OPENAI_API_KEY"] || "No key found",
});

const userInterface = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const askUser = async () => {
    return new Promise((resolve) => {
        userInterface.question('Spør chatbot: ', (input) => {
            resolve(input);
        });
    });
};

async function chatbot (input: string, chatHistory: chat[]): Promise<{response: string, chatHistory: chat[]}> {
    userInterface.prompt();


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


async function similarWiki(input: string) {
    const inputUserEmbed = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: input,
    })
    const questionVectorEmbedding = inputUserEmbed.data[0].embedding;

    let wiki_Embeds = await Promise.all(dataset.map(async (wiki) => {
        const QandA_Embed = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: wiki,
        })
        const questionEmbedding = QandA_Embed.data[0].embedding;
        return {embed: questionEmbedding, wiki }
    }))



    const similarityScores = wiki_Embeds.map((wiki) => {
        return {
            similarity: similarity(questionVectorEmbedding, wiki.embed),
            wiki: wiki.wiki
        }
    })

    similarityScores.sort((a, b) => b.similarity - a.similarity);

    return similarityScores[0].wiki
}


let chat: chat[]= [
    {
        role: "system",
        content: "Du er en hjelpsom assistent som svarer på sprøsmål bare med bruk av infromasjonene fra wikien nedefor. Hvis du ikke har svaret på et spørsmål ber du brukeren utdype spørsmålet eller si du ikke har svaret."
    }
]

while(true) {
    const input = await askUser() as string

    if (input === "exit") {
        userInterface.close();
        console.log("Goodbye!")
        console.log(chat)
        break;
    }

    const { response, chatHistory } = await chatbot(input, chat);

    console.log(response)
    chat = chatHistory
}


/* ----Gammel promt ------
Oppgaven din er å være en chatbot for bedriften Skyttel AS. Dette står på hjemmesiden dies:Skyttel et teknologidrevet selskap, og tilbyr ulike tjenester og løsninger for transaksjoner til transportsektoren. Virksomheten har i snart 70 år driftet betalingsløsninger og forskjellige bompengeprosjekter. I 1953 startet vi innkreving av bompenger for Puddefjordsbroen, og startet tre år senere også innkreving av bompenger for Eidsvågstunnelen.
                    
                    Du skal kunne svare på spørsmål om bedriftens tjenester som bompengeprosjekter, betalingsløsninger og transaksjoner til transportsektoren. Annet enn det så skal du også kunne sende pdf af faktura og annet. 

                    Hvis du ikke har svaret på et spørsmål så refererer du brukeren til kundesenteret på post@skyttel.no.
*/