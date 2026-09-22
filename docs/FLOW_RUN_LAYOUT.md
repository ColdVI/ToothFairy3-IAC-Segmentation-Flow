# Drive dosya haritası ve yeni koşu adları
22 Eylül 2026. Ana konum: ToothFairy/ToothFairy3.
Bu turda Drive'da dosya silinmedi veya taşınmadı. Büyük cache dizinlerindeki tüm hasta dosyaları tek tek indirilmedi ve kopya hash eşitliği doğrulanmadı. Koşullu adaylar doğrudan silme listesi değildir.

## Korunacak çekirdek
Orijinal dataset ve nnUNet_results yanında kesin splits.json, OOF provenance, seçili flow checkpoint'leri, koşu config'leri ve sonuç CSV/JSON'ları korunmalı. Cache yeniden hazırlama; checkpoint yeniden eğitim gerektirebilir.

| Dosya/klasör | Neye ait? | Öneri |
|---|---|---|
| ToothFairy3/ | Orijinal CBCT, labelsTr, dataset.json | Koru |
| iac_runs/nnUNet_results/ | Üç sınıflı nnU-Net; Dataset801_IAC_LR | Checkpoint, plans ve metadata ile koru |
| iac_runs/configs_cache/splits.json | Development/external ve fold sözleşmesi | Kesin koru |
| iac_runs/canalmanifold_oof_softmax/ | fold_0…4 ve external_singlefold olasılıkları | CMF/GeoFlow/IAC-B ortak girdisi; koruman önerilir |
| iac_runs/dataset_cache_colab_v2/ | Dönüştürülmüş Dataset801 | Şimdilik koru; IAC-B preflight buradaki dataset.json'ı okuyor |
| iac_runs/dataset_cache_colab_v1/ | Eski dönüştürülmüş veri | Kapsam/dönüşüm eşitliği doğrulanırsa büyük kopyalar kaldırılabilir; metadata kalsın |
| iac_runs/sdf_cache_backup/ | OOF probabilities/hard, GT/prior SDF ve oof_manifest.json | Tümünü silme; manifesti koru, tekrarı doğrula, SDF cache'lerini üretim tarifiyle ayır |
| iac_runs/outputs/ | Baseline, Prompt1/2/3R ve analiz sonuçları | Küçük CSV/JSON raporlarını arşivle |
| iac_runs/prompt3r_stage1a_v2/ | İlk dense-SDF düzeltmeleri, panel/tau/loss/smoke kayıtları | Rapor ve seçili checkpoint'i arşivle |
| iac_runs/prompt3r_stage1a_v3_overnight/ | Sonraki Prompt3R gece koşusu | Rapor, panel, log ve seçili checkpoint'i arşivle |
| iac_runs/code/ | apply_prompt3r_stage1a_fix.py ve find_inf_in_config.py | Özgün kaynak arşivi yapılmadan silme |
| iac_runs/old/ | flow_fold0 ve Fast=True eski koşuları | Seçili ağırlık/sonuçlar ayıklanmadan toplu silme |
| iac_runs/caches/ | Listeleme anında boş | Aktif kullanım yoksa kaldırılabilir |
| create_dataset504_iac_lr.py | Hocanın binary + yardımcı SDF/bridge kodu | Koru |
| teacher_baseline_results/ | Teacher ağırlıkları, snapshot ve 107-vaka değerlendirmesi | Ağırlık ve _full_fold0_evaluation raporlarını koru |
| teacher_baseline_cache/ | Teacher binary_raw / nnUNet_preprocessed | Metadata/split/tarif korunduktan sonra büyük cache kaldırılabilir |
| CanalManifoldFlow/ | İlk CMF kodu, corrected_v1.zip ve q0_migrate_compact.py | Eski özgün sürüm arşivi; v2 bire bir yedeği sayılmaz |
| CanalManifoldFlow_corrected_v1/ | Corrected CMF kod paketi | Bu tam sürüm ayrıca korunmalı |
| CanalManifoldFlow_v2_GeoFlow_Newton/ | CMF v2 / GeoFlow kod, config, notebook | Kod GitHub'a alındı; eski notebook yolları ve notebook kopyaları ayrıca korunmalı |
| canalmanifold/ | CMF/GeoFlow cache, runs, audits, external sonuçları | Kod değil; checkpoint/raporu cache'den ayır |
| IAC_Flow_Training_Package/ | Gaussian→SDF kaynakları ve runs/fold_0 | Kaynak GitHub'da; latest.pt / previous.pt ağırlıklarını ayrıca koru |
| iacb/ | Yeni stokastik SDF bridge kodu | Güncel Python paketi GitHub'a alındı; legacy/ ve eski notebook/planlar ayrıca korunmalı |
| iacb_runs_532_rpi_v2/ | RPI cache, eski eval ve yeni case-balanced koşu | best/last, epoch_split, epoch_log ve raporları koru |
| iacb_runs_532/ | Listeleme anında boş | Aktif notebook kullanmıyorsa kaldırılabilir |
| iacb_runs_532_aligned_v1/ | Listeleme anında boş | Aynı koşul |
| iac_flow_v1/ | Listeleme anında boş | İsminden uygulanmış mimari/deney sonucu çıkarılamaz |
| iac_flow_v1_runs/rectified_sdf_heun_v1/ | Listeleme anında boş | Aktif yazım olmadığını doğrulamadan silme |

## canalmanifold ayrıntısı
- cache/ ve cache_q0centered_v2/: ilk/CMF q0 temsilleri için yeniden üretilebilir önhesaplama.
- runs_q0centered_v2/ ve runs_corrected_phase_v1_1000/: seçili checkpoint, config ve exact sonuçlar korunmalı.
- audits_corrected_phase_v1/: küçük araştırma kanıtları, koru.
- external_tf3s_corrected_v1/: eski S-test protokolünün sonuçları, koru.
- manifest.json, colab_fold0.yaml ve config_q0centered_v2.yaml: koşularıyla beraber koru.
- v2/cache_m8_v2/: daha ayrıntılı temsil cache'i.
- v2/runs/exact_geoflow_newton_fold0/: summary.json ve metrics.csv korunmalı.
- v2/runs/fold_0/geoflow_newton/: seçili ağırlıklar korunmalı.
- v2/runs/oracle_fold0/: oracle/headroom raporları; normal tahmin sonucu diye sunulmamalı.
- v2/canalmanifold_geoflow_newton.yaml: ilgili koşunun config'i.

## En yeni IAC-B koşusu
iacb_runs_532_rpi_v2/epoch_run_v3_casebalanced içinde best.pt ve last.pt yaklaşık 36,8 MiB/adet. epoch_split.json ve epoch_log.jsonl ile koru. 45 epoch, en iyi patch-validation loss epoch 27; yeni full-volume değerlendirme bu dizinde bulunmadı.

fold_0_eval eski bir koşunun raporu: deterministik bridge çıktılarında Dice 0. Bu sonucu yeni case-balanced ağırlığa atama. cache/ içindeki RPI image/SDF/probability/GT/indeks/meta dizilerini tam hizalama ve hazırlama tarifi olmadan silme. epoch_run_v2 listelemede boş.

Gaussian-SDF runs/fold_0/latest.pt ve previous.pt yaklaşık 288 MiB/adet. Boyutlarının aynı olması içeriklerinin aynı olduğu anlamına gelmez.

## Yeni ad düzeni
Örnek:
20260922T090000Z__iacb-bridge__tf3-532-rpi-v2__fold-0__seed-42__casebalanced-v3

| Yeni konum | İçerik |
|---|---|
| runs/<deney-adı>/config/ | Tam resolved config ve kesin split kopyası |
| runs/<deney-adı>/checkpoints/ | best/last ve trainer'ın çıktıları |
| runs/<deney-adı>/metrics/<eval-adı>/ | CSV/JSON; kohort ve evaluator ayarları |
| runs/<deney-adı>/logs/ | Konsol logları |
| runs/<deney-adı>/predictions/ | İsteğe bağlı maske/görseller |
| cache/<temsil-ve-veri-sürümü>/ | Tekrar üretilebilir cache ve tarifi |
| archive/<eski-deney>/ | Seçili ağırlık + config + split + sonuç + kod sürümü |

Bu düzen bundan sonraki koşular içindir. Eski notebook'ların yol/adlarını otomatik değiştirmez. Mevcut CLI'larda --out/--work ve GeoFlow paths.run_dir değerleri açıkça yeni konuma ayarlanmalı.

## Temizlik sırası
1. Dataset, model, split, OOF/provenance'ı koru.
2. Her özgün deneyin seçili ağırlığı, config'i ve raporunu arşivle.
3. Eski kodun tam sürümünü koru; en yeni kod tüm eski sürümlerin bire bir yedeği değildir.
4. __pycache__, yeniden oluşan egg-info ve kullanılmayan boş dizinleri ayıkla.
5. Büyük tekrar SDF/cache/veri kopyalarına, kapsam ve geometri doğrulamasından sonra geç.
