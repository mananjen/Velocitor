# Failure Analysis

These three cases were selected from the evaluation runs to illustrate one representative failure mode from each retriever implemented:

- **BM25**: lexical / terminology mismatch
- **Dense-MiniLM**: semantic drift from a noisy, ambiguous query
- **Hybrid-RRF**: fusion failure, where combining sparse and dense rankings reinforces the wrong interpretation instead of correcting it

## Failure Case 1 — BM25

**Retriever:** `bm25`  
**Query ID:** `17`  
**Query:** Income tax exemptions for small business?

### Top-5 retrieved passages

1. **doc_id:** `276295` | **score:** `8.064651`  
   - "There's no homestead property tax exemption in TN. According to the TN comptroller site: Exemptions Exemptions are available for religious, charitable, scientific, and nonprofit educational uses, governmental property, and cemeteries. Most nongovernmental exemptions require a one-time application and approval by the State Board of Equalization (61..."

2. **doc_id:** `141458` | **score:** `7.507319`  
   - "Not really, no. The assumption you're making—withdrawals from a corporation are subject to ""[ordinary] income tax""—is simplistic. ""Income tax"" encompasses many taxes, some more benign than others, owing to credits and exemptions based on the kind of income. Moreover, the choices you listed as benefits in the sole-proprietor case—the RRSP, the ..."

3. **doc_id:** `22353` | **score:** `7.034221`  
   - "The short answer is you're tax exempt if the tax laws say you are. There are a bunch of specific exemptions based on who you are, what you're buying and why. Taking British Columbia as an example. One exemption is supplies for business use: Some exemptions are only available to certain purchasers in certain circumstances. These exemptions include:..."

4. **doc_id:** `531499` | **score:** `6.517993`  
   - "That $200 extra that your employer withheld may already have been sent on to the IRS. Depending on the size of the employer, withholdings from payroll taxes (plus employer's share of Social Security and Medicare taxes) might be deposited in the US Treasury within days of being withheld. So, asking the employer to reimburse you, ""out of petty cash..."

5. **doc_id:** `519473` | **score:** `6.494978`  
   - "The difference between the provincial/territorial low and high corporate income tax rates is clear if you read through the page you linked: Lower rate The lower rate applies to the income eligible for the federal small business deduction. One component of the small business deduction is the business limit. Some provinces or territories choose to u..."

### Gold passage

**Best-ranked gold doc_id:** `146657`  
**Gold rank in system:** `484`  
**Gold score in system:** `3.454126`  
**Gold passage snippet:** Yes, you should be able to deduct at least some of these expenses. For expense incurred before you started the business: What Are Deductible Startup Costs? The IRS defines “startup costs” as deductible capital expenses that are used to pay for: 1) The cost of “investigating the creation or acquisition of an active trade or business.” This includes ...

### Diagnosis

This is a lexical mismatch failure. BM25 overweights exact surface terms such as `income`, `tax`, and especially `exemptions`, so it ranks passages that literally discuss tax exemptions ahead of the gold passage. The gold answer is conceptually relevant, but it is phrased in a different finance/tax vocabulary: `deductible startup costs`, `capital expenses`, and `deductions` rather than `exemptions`. Since the current BM25 retriever is purely sparse and does not model semantic equivalence between related tax concepts, it misses the answer.

### Specific retrieval fixes

**Use Hybrid candidate generation for the ranking rather than just BM25 alone like requested as 3rd model style.**  


   The problem here is first-stage recall: the gold passage is at rank 484, so a sparse-only pipeline is not surfacing the right candidate set. A combined sparse+dense first stage would improve coverage before reranking, similar to what we have done. 

---

## Failure Case 2 — Dense-MiniLM

**Retriever:** `dense-minilm`  
**Query ID:** `60`  
**Query:** Can Health-Releated Services be a Business Expense?

### Top-5 retrieved passages

1. **doc_id:** `231449` | **score:** `0.624458`  
   - Doesn't matter what the product is, whether it's a tangible good or a service, a business is still a business and must be run as thus. If you don't run a hospital properly, as a business that provides a service, then that hospital is soon to be threatened with closure or state take-over.

2. **doc_id:** `82140` | **score:** `0.585424`  
   - "Just as with any other service provider - vote with your wallet. Do not go back to that doctor's office, and make sure they know why. It's unheard of that a service provider will not disclose the anticipated charges ahead of time. A service provider saying ""we won't tell you how much we charge"" is a huge red flag, and you shouldn't have been dea..."

3. **doc_id:** `510863` | **score:** `0.581124`  
   - No. The equipment costs are not necessarily a direct expense. Depending on the time of purchase and type of the expenditure you may need to capitalize it and depreciate it over time. For example, if you buy a computer - you'll have to depreciate it over 5 years. Some expenditures can be expensed under Section 179 rules, but there are certain condit...

4. **doc_id:** `295793` | **score:** `0.580588`  
   - In short, no, or not retroactively. There really are multiple companies involved, each of which bills you separately for the services they provided. This can be partly avoided by selecting either a high-end health plan with lower out-of-pocket maximum, (costs more up front, of course) or by selecting a genuine Health Management Organization (not a ...

5. **doc_id:** `451993` | **score:** `0.579895`  
   - Why the hell is health care still tied to employment? Does anyone really think that is a reasonable function of business? All this does is make hiring human workers less attractive. Guess what. Thanks to automation and outsourcing business owners increasingly have the choice to do without the hassle of hiring people, especially poor, unskilled peop...

### Gold passage

**Best-ranked gold doc_id:** `381151`  
**Gold rank in system:** `39`  
**Gold score in system:** `0.481812`  
**Gold passage snippet:** Chris, since you own your own company, nobody can stop you from charging your personal expenses to your business account. IRS is not a huge fan of mixing business and personal expenses and this practice might indicate to them that you are not treating your business seriously, and it should classify your business as a hobby. IRS defines deductible b...

### Diagnosis

This failure exposes a weakness of the current dense retriever design: it uses a **single-vector bi-encoder** (`all-MiniLM-L6-v2`) to compress the whole query into one embedding. That representation is brittle when the query is both noisy and ambiguous. Here, the typo in `Health-Releated` hurts lexical anchoring, and the phrase `health-related services` pushes the embedding toward the broad healthcare/service/business neighborhood. As a result, the retriever prefers passages about healthcare as an industry, medical service transactions, and generic service expenses, while the gold passage is actually about **tax deductibility versus personal expense classification**.

### Specific retrieval fixes

**Using a better Typo-robust tokenizer instead of the single-vector bi-encoder, or a second-stage interaction reranker over the top dense candidates.**

   A subword/character-aware sparse retriever is more robust to noisy tokenization and preserves important lexical anchors like `business expense`. (*Implementing the bge-base-en-v1.5 model should improve performance.*) 

   A cross-encoder or late-interaction reranker can use full query–passage token interactions, which is exactly what the current single-vector dense retriever lacks. That would help separate “healthcare/service” passages from passages about whether the cost is deductible as a business expense. (*e.g. cross-encoder/ms-marco-MiniLM-L-6-v2. Have to keep the latency in mind.*)

---

## Failure Case 3 — Hybrid-RRF

**Retriever:** `hybrid-rrf`  
**Query ID:** `77`  
**Query:** Can the IRS freeze a business Bank account?

### Top-5 retrieved passages

1. **doc_id:** `508754` | **score:** `0.031010`  
   - "I have checked with Bank of America, and they say the ONLY way to cash (or deposit, or otherwise get access to the funds represented by a check made out to my business) is to open a business account. They tell me this is a Federal regulation, and every bank will say the same thing. To do this, I need a state-issued ""dba"" certificate (from the co..."

2. **doc_id:** `364378` | **score:** `0.029211`  
   - As an LLC you are required to have a separate bank account (so you can't have one account and mix personal and business finances together as you could if you were a sole trader) - but there's no requirement for it to be a business bank account. However, the terms and conditions of most high street bank personal current accounts specifically exclude...

3. **doc_id:** `525200` | **score:** `0.028068`  
   - I wouldn't do this. There is a chance that your check could get lost/misdirected/misapplied, etc. Then you would need to deal with the huge bureaucracy to try to get it fixed while interest and penalties pile up. What you can do is have the IRS withdraw the money themselves by providing the rounting number and account number of your bank. This shou...

4. **doc_id:** `384192` | **score:** `0.026861`  
   - Technically, it's only when you need to pass money through. However consider that the length the account has been open builds history with the financial institution, so I'd open ASAP. Longer history with the bank can help with getting approved for things like business credit lines, business cards, and other perks, though if you're not making money ...

5. **doc_id:** `155389` | **score:** `0.026646`  
   - And if you need to pay business taxes outside of the regular US 1040 form, you can use the IRS' Electronic Federal Tax Payment System (EFTPS). Basically, you enroll your bank accounts, and you can make estimated, penalty, etc. payments. The site can be found here.

### Gold passage

**Best-ranked gold doc_id:** `551315`  
**Gold rank in system:** `71`  
**Gold score in system:** `0.010417`  
**Gold passage snippet:** If the business is legally separated and not commingled - they probably cannot. What they can do is put a lien on it (so that you cannot sell the business) and garnish your income. If the corporate veil is pierced (and its not that hard to have it pierced if you're not careful) - then they can treat it as if it is your personal asset. Verify this w...

### Diagnosis

This is a fusion failure. The current hybrid retriever uses **fixed, equal-weight RRF** over BM25 and dense candidates. In this case, both branches are attracted to shallow signals like `IRS`, `business`, `bank account`, and `business account`, so RRF reinforces a common but incorrect interpretation of the query as a general banking/account-management problem. The gold passage is different: it answers the legal-enforcement question in terms of **liens, garnishment, asset separation, and piercing the corporate veil**. The current fusion rule cannot distinguish “both retrievers agree on something” from “both retrievers agree on the wrong thing.”

### Specific retrieval fixes

**Replace fixed equal-weight RRF with weighted or query-adaptive fusion. Or adding a reranker over the hybrid.**  


   Equal fusion with BM25 is likely dragging the ranking toward lexical false positives. we can use a better weighted fusion score. (*e.g. fused_score = α * dense_norm + (1-α) * bm25_norm with α ~ 0.6 or 0.7*) 

   RRF does not reason about which interpretation of the query is actually correct. A reranker over the fused candidate set would give better results as mentioned in the dense improvement also. (*e.g. cross-encoder/ms-marco-MiniLM-L-6-v2*)