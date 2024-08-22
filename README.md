# RAG-MODEL

Öncelikle log dosyaları belirli bir formatta olduğu için bu verileri düzenleme işlemi yapıldı.
Pinecone hesabı açılarak bir API key alındı. Verileri vectör veritabanına aktarabilmek için chunklara bölündü ver her chunk 200 boyutunda seçildi. 
Pinecone yüklenmesi için veriler vectörlere dönüştürüldü. Gerekli ön işleme adımları yapıldıktan sonra veriler Pinecone veritabanına yüklendi.


![image](https://github.com/user-attachments/assets/e0fbbca2-0591-4039-a933-d44cba2eea87)

Daha sonra gpt dil modeli kullanıldı ve veriler vectör veritabanından gelen eşleşme ve gpt den gelen yanıtla birleşrek bir cevap oluşturulmaya çalışıldı.
Ancak cevap konusunda başarılı olunamadı. İki ayrı cevap üretiyor sistem. Örnek soru:
-------------------
Q: SHOW DATA FROM 17/May/2015
A: 
Context:
08 | Date: 17/May/2015:17:05:11 | File Extension: nagios-sms4.png | Status: 200 | Size: 200 | OS: KHTML, like Gecko
IP: 122.166.142.108 | Date: 17/May/2015:17:05:08 | File Extension: pc-load-letter.jp 18/May/2015:01:05:47 | File Extension: computer-keyboard-jacket.jpg | Status: 200 | Size: 200 | OS: KHTML, like Gecko
IP: 86.76.247.183 | Date: 18/May/2015:01:05:22 | File Extension: notes.js | Status : 17/May/2015:17:05:36 | File Extension: kibana-logstash-downloads.png | Status: 200 | Size: 200 | OS: KHTML, like Gecko
IP: 122.166.142.108 | Date: 17/May/2015:17:05:35 | File Extension: kibana-chef- 2015:18:05:36 | File Extension: puppet-at-loggly.pdf.html | Status: 200 | Size: 200 | OS: KHTML, like Gecko
IP: 70.190.77.28 | Date: 17/May/2015:18:05:39 | File Extension: favicon.ico | Status: 200 |  7/May/2015:17:05:49 | File Extension: Unknown File | Status: 200 | Size: 200 | OS: Unknown OS
IP: 122.166.142.108 | Date: 17/May/2015:17:05:09 | File Extension: office-space-printer-beat-down-gif.gif 

Question: SHOW DATA FROM 17/May/2015
Answer: Yes, this is a very simple question.
The answer is: "Show data from 17.05.2015 to 17."
This is the same question as the one above. The question is, "What is this data?"
I have a question for you. I am a programmer. What is your question?
You can answer it in the following way: 1. Show data to the user. 2. Use the data. 3. Send the request to your server. 4. Upload the file to a server that you want to use. 5. Download the files. 6. Share the results. 7. Enjoy the result. 8. If you like, you can share the link to this page. 9. You can also

![image](https://github.com/user-attachments/assets/85d09d7a-be68-4d24-b3f0-58e99a3c4e87)




