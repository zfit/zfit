# Namescopes

currently, if we load a parameter and it exists already in the namespace, this will be used.
That can be very dangerous for automatically created parameters, such as autoparams in SumPDFs.

# recursive replacement

need to replace also all the PDFs, data for the loss

# unique names

Every object should have a unique name. This is not the case for the parameters, which are created automatically. This is not a problem for the user, but for the serialization.
We also need to capture recursively all the names of stacked objects such as PDFs in a SumPDF.
