import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from train import *

model_trained = AutoModelForSeq2SeqLM.from_pretrained('/content/drive/MyDrive/title_generation_final/model-t5-base/checkpoint-5600/').to('cuda')
token_trained = AutoTokenizer.from_pretrained('/content/drive/MyDrive/title_generation_final/model-t5-base/checkpoint-5600/')

temperature = 0.9
num_beams = 4
max_gen_length = 128

text='''(Reuters) - U.S. President Donald Trump faces an temporary hold on his travel ban on seven mainly Muslim countries, but the outcome of a ruling on the executive order’s ultimate legality is less certain.  Any appeals of decisions by U.S. District Court Judge James Robart in Seattle face a regional court dominated by liberal-leaning judges who might not be sympathetic to Trump’s rationale for the ban, and a currently shorthanded Supreme Court split 4-4 between liberals and conservatives. The temporary restraining order Robart issued on Friday in Seattle, which applies nationwide, gives him time to consider the case in more detail, but also sends a signal that he is likely to impose a more permanent injunction.    The Trump administration has appealed that order. The San Francisco-based 9th U.S. Circuit Court of Appeals said late on Saturday that it would not decide whether to lift the judge’s ruling, as requested by the U.S. government, until it receives briefs from both sides, with the administration’s filing due on Monday. Appeals courts are generally leery of upending the status quo, which in this case - for now - is the suspension of the ban.  The upheaval prompted by the new Republican administration’s initial announcement of the ban on Jan. 27, with travelers detained at airports upon entering the country, would potentially be kickstarted again if Robart’s stay was lifted. The appeals court might also take into account the fact that there are several other cases around the country challenging the ban. If it were to overturn the district court’s decision, another judge somewhere else in the United States could impose a new order, setting off a new cascade of court filings.  If the appeals court upholds the order, the administration could immediately ask the U.S. Supreme Court to intervene. But the high court is generally reluctant to get involved in cases at a preliminary stage, legal experts said. The high court is short one justice, as it has been for a year, leaving it split between liberals and conservatives. Any emergency request by the administration would need five votes to be granted, meaning at least one of the liberals would have to vote in favor.  “I think the court’s going to feel every reason to stay on the sidelines as long as possible,” said Steve Vladeck, a professor at the University of Texas School of Law. Trump last week nominated a conservative appeals court judge, Neil Gorsuch, to fill the vacancy, but he will not be sitting on the Supreme Court for at least two months. Gorsuch’s vote, if he is confirmed by the U.S. Senate, could come into play if the case were to reach the court at a later stage of the litigation. Once the case proceeds past the injunction stage of the litigation and onto the merits of whether the order is legally sound, legal experts differ over how strong the government’s case would be.  Richard Primus, a professor of constitutional law at the University of Michigan Law School, said the administration could struggle to convince courts that the ban was justified by national security concerns.  The Supreme Court has previously rejected the idea that the government does not need to offer a basis for its actions in the national security context, including the landmark 1971 Pentagon Papers case, in which the administration of President Richard Nixon tried unsuccessfully to prevent the press from publishing information about United States policy toward Vietnam. “The government’s argument so far in support of the order is pretty weak,” Primus said. Jonathan Adler, a professor at Case Western Reserve University School of Law, said the administration has legal precedent on its side, with the courts generally deferential to executive action on immigration.  However, he said it is unusual for the courts to be asked to endorse “a policy that appears to have been adopted in as kind of haphazard and arbitrary way as this one appears to have been.” '''
inputs = token_trained([text], max_length=512, return_tensors='pt',truncation=True)

title_ids = model_trained.generate(
    inputs['input_ids'].to('cuda'), 
    num_beams=num_beams, 
    temperature=temperature, 
    max_length=max_gen_length, 
    early_stopping=True
)
title = token_trained.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(title)
