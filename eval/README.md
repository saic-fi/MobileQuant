<div align="center">

## Evaluating the pre-quantized LLM models

</div>

The table below includes the checkpoints for each models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Quantization</th>
<th valign="bottom">CKPT</th>  
<th valign="bottom">WikiText </th>
<th valign="bottom">ARC-C </th>
<th valign="bottom">Hellaswag </th>
<th valign="bottom">MMLU </th>
<!-- TABLE BODY -->
<tr>
      <td align="left">TinyLlaMA-1.1B-v1.0-Chat</td>
      <td align="center">W8A8</td>
      <td align="center"><a href=https://huggingface.co/fwtan/llama-1.1b-mobilequant-w8a8-s1024-e60-hf>ckpt</td>
      <td align="center"> 15.5 </td>
      <td align="center"> 31.9 </td>
      <td align="center"> 59.2 </td>
      <td align="center"> 25.0 </td>
</tr>
<tr>
      <td align="left">TinyLlaMA-1.1B-v1.0-Chat</td>
      <td align="center">W4A8</td>
      <td align="center"><a href=https://huggingface.co/fwtan/llama-1.1b-mobilequant-w4a8-s1024-e60-hf>ckpt</td>
      <td align="center"> 17.1 </td>
      <td align="center"> 32.3 </td>
      <td align="center"> 57.0 </td>
      <td align="center"> 25.5 </td>
</tr>
<tr>
      <td align="left">StableLM-2-1.6B</td>
      <td align="center">W8A8</td>
      <td align="center"><a href=https://huggingface.co/fwtan/stablelm-2-1_6b-mobilequant-w8a8-s1024-e60-hf>ckpt</td>
      <td align="center"> 29.7 </td>
      <td align="center"> 37.1 </td>
      <td align="center"> 63.6 </td>
      <td align="center"> 30.0 </td>
</tr>
<tr>
      <td align="left">StableLM-2-1.6B</td>
      <td align="center">W4A8</td>
      <td align="center"><a href=https://huggingface.co/fwtan/stablelm-2-1_6b-mobilequant-w4a8-s1024-e60-hf>ckpt</td>
      <td align="center"> 33.6 </td>
      <td align="center"> 35.6 </td>
      <td align="center"> 60.5 </td>
      <td align="center"> 24.1 </td>
</tr>
<tr>
      <td align="left">Gemma-2B</td>
      <td align="center">W8A8</td>
      <td align="center"><a href=https://huggingface.co/fwtan/gemma-2b-mobilequant-w8a8-s1024-e60-hf>ckpt</td>
      <td align="center"> 20.3 </td>
      <td align="center"> 21.8 </td>
      <td align="center"> 40.9 </td>
      <td align="center"> 25.8 </td>
</tr>
<tr>
      <td align="left">Gemma-2B</td>
      <td align="center">W4A8</td>
      <td align="center"><a href=https://huggingface.co/fwtan/gemma-2b-mobilequant-w4a8-s1024-e60-hf>ckpt</td>
      <td align="center"> 21.4 </td>
      <td align="center"> 23.0 </td>
      <td align="center"> 38.9 </td>
      <td align="center"> 25.6 </td>
</tr>
</tbody></table> 

## Running the evaluation

- Download the checkpoint

```
CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --tasks wikitext,arc_challenge,hellaswag,hendrycksTest* --mode custom --hf_path ${CKPT} --output_dir ${OUTPUT_DIR}
```
