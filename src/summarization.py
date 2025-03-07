from transformers import BertTokenizerFast, EncoderDecoderModel

huggingface_model = 'mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization'
tokenizer = BertTokenizerFast.from_pretrained(huggingface_model)
model = EncoderDecoderModel.from_pretrained(huggingface_model)

#Input paragraph 

paragraph = """
I - Identity Target Users. 
Think about all the players in the ecosystem. For example, user segmentation for a marketplace like Doordash consists of restaurants, buyers, and drivers. For the FB marketplace, the users segment includes not only sellers and buyers but also advertisers. Candidates often miss this edge case of dashers or advertisers when defining the possible user segments. 
Go one level deeper. Broad segmentation is not enough. If you are talking about buyers, go one level deeper into different types of buyers. 
Behavioral segmentation is always better than demographic segmentation. For example, it's easy to come up with kids, teens, adults as segmentation or using male female segmentation but that's too generic and won't make you stand out. Think about segmentation based on income group, frequency of use, and segmentation with the most pressing pain points to the least. For example, a trading app can segment users to beginners to investing, some experienced (general stocks) to advanced( stocks, future options), etc.  
Always think of MECE segmentation: Mutually exclusive and collectively exhaustive segmentation. 
For example, mothers and working professionals are not mutually exclusive, as not all mothers are stay-at-home moms, and there are working moms, too. 
Segment Prioritization: Once you have segmented your users, you need to prioritize it. Remember, your goal as a PM is to take calculated assumptions and de-risk the riskiest assumption as fast as possible. Choose a segment where you can get the biggest impact, either a large enough segment to get results ASAP or high paying segment to get revenue faster. Again, your segmentation strategy ties back to your goal. Are you improving adoption, ARPU, revenue, or retention? This further highlights why defining a mutually agreeable goal is important. 
Some criteria to segment 
TAM: Total addressable market tells you how large a segment is. All other things being equal, a large segment will give you data faster and is a potential for bigger revenue unless you are building a solution for a specific niche. 
Frequency of use: The more frequently a segment uses your product, the more valuable they are. You will get your data faster, and you may have a higher retention rate. 
Spending Power: Use this criteria if your goal is to increase ARPU. 
Underserved users: If you find a large enough niche of users whose needs are unmet by existing solutions, you have a competitive advantage against existing players. Use this criteria only if you feel the problem space has the potential to impact some underserved users. 


"""

#Tokenize the paragraph

inputs = tokenizer([paragraph], 
                   padding= "max_length",
                   truncation=True,
                   max_length=256,
                   return_tensors="pt")

#Define the input IDs and attentiion mask 
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask 

#Generate the output
output = model.generate(input_ids, attention_mask=attention_mask)

#Decode the output
decode_output = tokenizer.decode(output[0], skip_special_tokens=True)

#print the summary
print(decode_output)

